"""Private HTTP connection to the copied native driver process; no drivetrain calculations."""
import json
from pathlib import Path
import signal
import subprocess
import time
import httpx

ROOT=Path(__file__).resolve().parent

class NativeDriver:
    def __init__(self):
        self.process=None
        self.http=httpx.Client(transport=httpx.HTTPTransport(uds=str(ROOT/'native.sock')),
                               base_url='http://native',timeout=.25,trust_env=False)

    def start(self):
        (ROOT/'native.sock').unlink(missing_ok=True)
        with (ROOT/'native-driver.log').open('ab') as log:
            self.process=subprocess.Popen(['/bin/bash',str(ROOT/'start-native.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'native-driver-active.json').write_text(json.dumps({'pid':self.process.pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
        deadline=time.monotonic()+12
        error=None
        while time.monotonic()<deadline:
            if self.process.poll() is not None:raise RuntimeError('Native driver exited; inspect native-driver.log')
            try:return self.status()
            except httpx.TransportError as exc:error=exc;time.sleep(.05)
        raise TimeoutError('Native driver startup: '+str(error))

    def status(self):
        result=self.http.get('/health');result.raise_for_status();return result.json()

    def command(self,v,w):
        result=self.http.post('/command',json={'v':v,'w':w},timeout=.35)
        result.raise_for_status();return result.json()

    def stop(self):
        result=self.http.post('/stop',timeout=.35);result.raise_for_status();return result.json()

    def close(self):
        error=None
        if self.process is not None and self.process.poll() is None:
            try:self.stop()
            except Exception as exc:error=exc
            self.process.send_signal(signal.SIGTERM)
            try:self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait(timeout=2)
                error=RuntimeError('Native driver required SIGKILL; physical stop unconfirmed')
        self.http.close()
        (ROOT/'native-driver-active.json').unlink(missing_ok=True)
        if error:raise RuntimeError('Native shutdown: '+str(error))
