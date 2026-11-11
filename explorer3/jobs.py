"""Single subprocess trainer queue. Fixed commands only; no shell from browser."""
import json
import subprocess
import sys
import threading
import time
import uuid
from . import catalog
from .paths import DATA,ROOT,offline_environment
from .profiles import profile
from .resources import ResourceLease

class Jobs:
    def __init__(self, control):
        self.control=control
        self.process=None
        self.thread=None
        self.lock=threading.Lock()

    def enqueue(self, name: str, policy: str, smoke=True):
        profile(name)
        if policy not in {'act','smolvla'}:
            raise ValueError('Unknown trainer')
        if policy=='smolvla' and name!='so101_pick_place':
            raise ValueError('SmolVLA подготовлена только для SO-101')
        if policy=='smolvla' and not (DATA/'validation/smolvla-pass.json').exists():
            raise RuntimeError('SmolVLA ещё не прошла forward/backward/reload')
        episodes=[e for e in catalog.rows('episodes') if e['profile']==name and not e['excluded']
                  and e['category'] in {'demonstration','correction'} and e['outcome']=='success']
        if len(episodes)<3:
            raise RuntimeError('Нужны минимум три настоящих проверенных показа')
        if not smoke and len(episodes)<50:
            raise RuntimeError('Первичный ориентир — 50 качественных показов; завершите сбор')
        ident=uuid.uuid4().hex
        cfg={'profile':name,'policy':policy,'smoke':bool(smoke),'episodes':episodes,'seed':1000}
        with catalog.connect() as db:
            db.execute('INSERT INTO jobs(id,profile,policy,status,config,created) VALUES(?,?,?,?,?,?)',
                       (ident,name,policy,'queued',json.dumps(cfg),time.time()))
        return ident

    def tick(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            if self.control.owner is not None:
                return
            queued=[j for j in reversed(catalog.rows('jobs')) if j['status']=='queued']
            if queued:
                self.thread=threading.Thread(target=self._run,args=(queued[0],),daemon=True)
                self.thread.start()

    def _run(self, job):
        ident=job['id']
        try:
            with ResourceLease('compute'):
                if self.control.owner is not None:
                    raise RuntimeError('Physical control is active')
                path=DATA/'jobs'/f'{ident}.json'
                path.write_text(job['config'])
                with catalog.connect() as db:
                    db.execute("UPDATE jobs SET status='running' WHERE id=?",(ident,))
                with (DATA/'logs'/f'{ident}.log').open('a') as log:
                    self.process=subprocess.Popen([sys.executable,'-m','explorer3.training','--job',str(path)],
                        cwd=ROOT,env=offline_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=self.process.wait()
                status='completed' if code==0 else 'failed'
                checkpoint=None
                result_path=DATA/'jobs'/ident/'result.json'
                if code==0 and result_path.exists():
                    checkpoint=json.loads(result_path.read_text())['checkpoint']
                with catalog.connect() as db:
                    db.execute('UPDATE jobs SET status=?,error=?,checkpoint=? WHERE id=?',
                        (status,None if code==0 else f'Trainer exit={code}; см. журнал {ident}',checkpoint,ident))
        except Exception as exc:
            catalog.event('error',f'Trainer {ident}: {exc}')
            with catalog.connect() as db:
                db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",(str(exc),ident))
        finally:
            self.process=None

    def cancel(self):
        if self.process and self.process.poll() is None:
            import os,signal
            os.killpg(self.process.pid,signal.SIGTERM)
