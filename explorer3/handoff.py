"""Explicit user handover of SO-101; old code/configuration is never modified."""
from contextlib import ExitStack
import json
import os
import subprocess
import time
import urllib.request
from .paths import DATA
from .resources import ResourceLease, assert_hardware_available
from . import fleet_config

SERVICE='so101-ai-learning-ui.service'


def systemctl(*args,check=True):
    env={**os.environ,'XDG_RUNTIME_DIR':f'/run/user/{os.getuid()}',
         'DBUS_SESSION_BUS_ADDRESS':f'unix:path=/run/user/{os.getuid()}/bus'}
    return subprocess.run(['systemctl','--user',*args],env=env,text=True,capture_output=True,check=check,timeout=20)


def legacy_ready(timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8020/',timeout=1) as response:
                if response.status==200:return True
        except OSError:pass
        time.sleep(.2)
    return False


def idle(control):
    if control.owner is not None:raise RuntimeError('Сначала STOP и завершение сессии')


def leases(stack):
    for name in ('hardware','cameras','so101-leader'):
        stack.enter_context(ResourceLease(name))


def enter(control,paths,*,confirmed=False):
    if confirmed is not True:
        raise PermissionError('Подтвердите остановку старой системы :8020 до возврата')
    idle(control)
    with ResourceLease('so101-handoff'), ExitStack() as stack:
        leases(stack)
        previous=fleet_config.load(DATA)['robots']['so101']
        if not previous['enabled']:raise PermissionError('Включите SO-101 в допуске парка')
        active=systemctl('is-active',SERVICE,check=False).returncode==0
        saved=DATA/'handoff.json'
        state=json.loads(saved.read_text()) if saved.exists() else {}
        if state.get('restored',True):
            state={'service':SERVICE,'was_active':active,'restored':False,'started_at':time.time()}
        if state.get('service')!=SERVICE:raise ValueError('Invalid saved service name')
        fleet_config.atomic_write(saved,state)
        try:
            if active:systemctl('stop',SERVICE)
            if systemctl('is-active',SERVICE,check=False).returncode==0:
                raise RuntimeError('Служба :8020 не остановилась; проверьте её состояние')
            assert_hardware_available(paths)
            fleet_config.update_robot('so101',owner='explorer',root=DATA)
        except BaseException:
            if active:systemctl('start',SERVICE)
            state['restored']=active
            fleet_config.atomic_write(saved,state)
            raise
        control.mode='Explorer 3.0'
        return {'owner':'explorer','motion_started':False}


def restore(control,*,close_devices=None):
    idle(control)
    with ResourceLease('so101-handoff'):
        # Revoke new opens before closing existing readers. A failed return stays revoked.
        fleet_config.update_robot('so101',owner='legacy_8020',root=DATA)
        if close_devices is not None:close_devices()
        with ExitStack() as stack:
            leases(stack)  # refuses service startup while an Explorer handle remains open
            systemctl('start',SERVICE)
            if not legacy_ready():
                raise RuntimeError('Служба запущена, но :8020 не отвечает; повторите возврат после проверки службы')
            path=DATA/'handoff.json'
            state=json.loads(path.read_text()) if path.exists() else {'service':SERVICE}
            state.update(restored=True,returned_at=time.time())
            fleet_config.atomic_write(path,state)
        control.mode='Прежнее управление'
        return {'owner':'legacy_8020','http_8020':True}
