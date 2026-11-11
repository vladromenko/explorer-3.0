"""Only fixed SSH actions against the prepared Tank helper; no browser shell input."""
import json
import subprocess


def invoke(action):
    if action not in {'status','enter','restore'}:raise ValueError('Invalid handoff action')
    remote=['sudo','-n']
    if action!='status':
        # The restoration must survive a transient SSH loss when stock Wi-Fi restarts.
        remote+=['systemd-run','--quiet','--wait','--pipe','--collect','--unit=explorer3-tank-handoff']
    remote+=['/usr/bin/python3','/home/pi/explorer3/control/handoff.py',action]
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','hiwonder-tank',*remote],capture_output=True,text=True,timeout=65)
    if result.returncode:raise RuntimeError('Tank: '+result.stderr.strip())
    return json.loads(result.stdout)


def enter(control):
    if not control.authorized() or control.owner is not None:
        raise PermissionError('Нужны разрешённый физический блок Tank и STOP')
    return invoke('enter')


def restore(control):
    if control.owner is not None:raise RuntimeError('Сначала STOP и завершение сессии')
    result=invoke('restore')
    control.mode='Прежнее управление'
    return result
