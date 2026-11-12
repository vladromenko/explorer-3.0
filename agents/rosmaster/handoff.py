"""Temporary ownership of the native serial port/camera; originals are never edited."""
import json
import subprocess
from pathlib import Path

SERVICES=('rosmaster-console.service','explorer-control.service','explorer-watchdog.service',
          'rosmaster-bridge.service','explorer-camera-cache.service','explorer-camera.service')
CAMERA='explorer3-rosmaster-camera.service'


def unit(action,*names):
    return subprocess.run(['systemctl','--user',action,*names],capture_output=True,text=True,timeout=35)


def active(name):return unit('is-active',name).returncode==0


def checked(action,*names):
    result=unit(action,*names)
    if result.returncode:raise RuntimeError(result.stderr.strip() or 'systemctl '+action+' failed')


class Handoff:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'handoff-state.json'
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def status(self):
        saved=json.loads(self.path.read_text()) if self.path.exists() else None
        return {'saved':saved,'own_camera':active(CAMERA),
                'original_services':{name:active(name) for name in SERVICES}}

    def enter(self):
        if self.path.exists():
            saved=json.loads(self.path.read_text())
            if saved['boot_id']==self.boot:raise RuntimeError('Сначала вернуть прежнее управление после предыдущей передачи')
            self.path.rename(self.root/'handoff-previous-boot.json')
        saved={'boot_id':self.boot,'active':{name:active(name) for name in SERVICES}}
        self.path.write_text(json.dumps(saved,indent=2))
        try:
            # Console ExecStopPost starts control/watchdog, so stop those afterwards.
            checked('stop',SERVICES[0])
            checked('stop',*SERVICES[1:])
            if any(active(name) for name in SERVICES):raise RuntimeError('Прежний владелец не остановлен')
            owners=subprocess.run(['fuser','/dev/ttyUSB0'],capture_output=True,text=True,timeout=3)
            if owners.returncode not in (0,1):raise RuntimeError('Невозможно проверить владельца serial')
            if owners.stdout.strip():raise RuntimeError('Serial занят: '+owners.stdout.strip())
            checked('start',CAMERA)
        except Exception:
            self.restore()
            raise
        return saved

    def restore(self):
        checked('stop',CAMERA)
        if not self.path.exists():return {'restored':False,'reason':'No saved ownership'}
        saved=json.loads(self.path.read_text())
        if saved['boot_id']!=self.boot:raise RuntimeError('Снимок от другой загрузки; службы не изменены')
        # Restore bridge and camera first. Reconcile inactive services as well.
        for name in reversed(SERVICES):
            if saved['active'][name]:checked('start',name)
        for name in SERVICES:
            if not saved['active'][name]:checked('stop',name)
        actual={name:active(name) for name in SERVICES}
        if actual!=saved['active']:raise RuntimeError('Не все прежние службы восстановлены; снимок сохранён')
        self.path.rename(self.root/'handoff-last-restored.json')
        return {'restored':True,'services':actual}


if __name__=='__main__':
    import os
    print(json.dumps(Handoff(os.environ.get('EXPLORER3_ROOT','/home/vlad/explorer3')).restore()))
