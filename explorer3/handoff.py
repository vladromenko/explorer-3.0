"""Preserve exact previous service state; physical acceptance is performed in B."""
import json
import os
import subprocess
from .paths import DATA
from .resources import owners

SERVICE='so101-ai-learning-ui.service'

def systemctl(*args,check=True):
    env={**os.environ,'XDG_RUNTIME_DIR':f'/run/user/{os.getuid()}',
         'DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{os.getuid()}/bus'}
    return subprocess.run(['systemctl','--user',*args],env=env,text=True,capture_output=True,check=check)

def enter(control,paths):
    if not control.authorized() or control.owner is not None:
        raise PermissionError('Требуется разрешённый физический блок и STOP')
    active=systemctl('is-active',SERVICE,check=False).returncode==0
    (DATA/'handoff.json').write_text(json.dumps({'service':SERVICE,'was_active':active,'restored':False}))
    if active:
        systemctl('stop',SERVICE)
    held=owners(paths)
    if held:
        restore(control)
        raise RuntimeError(f'Другие владельцы устройств: {held}')

def restore(control):
    if control.owner is not None:
        raise RuntimeError('Сначала STOP и завершение сессии')
    path=DATA/'handoff.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state['service']!=SERVICE:
            raise ValueError('Invalid saved service name')
        if state['was_active'] and not state['restored']:
            systemctl('start',SERVICE)
        state['restored']=True
        path.write_text(json.dumps(state))
    control.mode='Прежнее управление'
