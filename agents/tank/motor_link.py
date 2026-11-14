"""Recoverable STOP latch and read-only udev monitoring, never motor commands."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def controller_present():
    for port in Path('/sys/class/tty').glob('ttyACM*'):
        usb=port/'device/..'
        try:
            if (usb/'idVendor').read_text().strip()=='1a86' and (usb/'idProduct').read_text().strip()=='55d4':return True
        except OSError:pass
    return False


class LinkState:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'motor-link-state.json'
        self.pending=False;self.reason=None;self.recovered=False;self.zero_writes=0;self.failures=[]
        if self.path.exists():
            state=json.loads(self.path.read_text())
            self.pending=bool(state['pending']);self.reason=state.get('reason');self.failures=state.get('failures',[])
        # A previous process's successful writes are not proof of today's link.

    def save(self):
        value={'pending':self.pending,'reason':self.reason,'failures':self.failures,
            'recovered':self.recovered,'zero_writes':self.zero_writes,'updated_at':time.time()}
        temporary=self.path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        temporary.replace(self.path)

    def lost(self,reason):
        if not self.pending:
            self.failures.append(time.time())
            self.failures=self.failures[-10000:]
        self.pending=True;self.reason=str(reason);self.recovered=False;self.zero_writes=0;self.save()

    def neutral_restored(self,count):
        if count!=3:raise ValueError('Required three completed neutral writes')
        self.recovered=True;self.zero_writes=count;self.save()

    def acknowledge(self,stationary):
        if stationary is not True:raise PermissionError('Подтвердите, что Tank стоит')
        if not self.recovered or self.zero_writes!=3:raise RuntimeError('Связь с моторной платой ещё не восстановлена')
        self.pending=False;self.reason=None;self.save()

    def status(self):
        now=time.time();today=datetime.datetime.now(datetime.timezone.utc).date()
        recent=sum(now-stamp<=600 for stamp in self.failures)
        return {'pending':self.pending,'recovered':self.recovered,'neutral_writes':self.zero_writes,
            'reason':self.reason,'last_failure_at':self.failures[-1] if self.failures else None,
            'failures_10_minutes':recent,'frequent_failures':recent>3,
            'failures_today':sum(datetime.datetime.fromtimestamp(stamp,datetime.timezone.utc).date()==today for stamp in self.failures),
            'physical_stop_confirmed':False}


class UdevMonitor:
    def __init__(self,on_removed):
        self.callback=on_removed;self.process=None;self.thread=None;self.error=None

    def start(self):
        try:
            self.process=subprocess.Popen(['udevadm','monitor','--udev','--property','--subsystem-match=tty'],
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
            self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
        except OSError as exc:self.error=str(exc)

    def read(self):
        properties={}
        for line in self.process.stdout:
            line=line.strip()
            if not line:
                usb=(properties.get('ID_VENDOR_ID')=='1a86' and properties.get('ID_MODEL_ID')=='55d4')
                tty=properties.get('DEVNAME','').startswith('/dev/ttyACM')
                if properties.get('ACTION')=='remove' and (usb or tty):self.callback()
                properties={}
            elif '=' in line:
                key,value=line.split('=',1);properties[key]=value

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            if self.thread:self.thread.join(timeout=1)
