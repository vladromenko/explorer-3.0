"""One managed compute worker, fixed commands only, explicit resume after interruption."""
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from . import catalog
from .datasets import freeze
from .paths import DATA, ROOT, offline_environment
from .profiles import profile
from .resources import ResourceLease, ResourceBusy, ExistingTrainingLease


def resumable_checkpoint(ident):
    candidates = []
    for root in (DATA / 'checkpoints').glob(ident + '-b*'):
        checkpoint = root / 'checkpoints/last/pretrained_model'
        if (checkpoint / 'train_config.json').is_file() and (checkpoint.parent / 'training_state').is_dir():
            candidates.append(checkpoint)
    if not candidates:
        raise ValueError('Нет сохранённого состояния optimizer; можно создать новое задание')
    return max(candidates, key=lambda p: p.stat().st_mtime)


class Jobs:
    def __init__(self, control):
        self.control = control
        self.process = None
        self.thread = None
        self.active_id = None
        self.cancel_requested = threading.Event()
        self.lock = threading.RLock()

    def enqueue(self, name, policy, smoke=True):
        profile(name)
        if policy not in {'act', 'smolvla'}:
            raise ValueError('Unknown trainer')
        if policy == 'smolvla' and (name != 'so101_pick_place' or not (DATA/'validation/smolvla-pass.json').exists()):
            raise ValueError('SmolVLA проверена только для SO-101')
        plan_id, _ = freeze(name, smoke)
        return self._insert(name, policy, {'kind': 'train', 'profile': name, 'policy': policy,
            'smoke': bool(smoke), 'plan_id': plan_id, 'seed': 1000})

    def sim_ppo(self, name):
        if name!='so101_pick_place_sim':
            raise ValueError('Среда ещё не принята')
        budget=json.loads((ROOT/'configs/sim-budgets.json').read_text())[name]['pick_place_ppo']
        return self._insert(name,'ppo',{'kind':'sim_ppo','profile':name,'steps':budget['timesteps'],
            'seed':budget['seed'],'device':'cpu','domain':'sim'})

    def evaluate(self, job_id, partition="validation"):
        if partition not in {"validation","test"}:raise ValueError("Unknown evaluation split")
        source = self.get(job_id)
        if source['status'] != 'completed' or not source['checkpoint']:
            raise ValueError('Сначала выберите завершённое обучение с checkpoint')
        cfg = json.loads(source['config'])
        if cfg.get('kind', 'train') != 'train':
            raise ValueError('Выберите задание обучения')
        return self._insert(source['profile'], source['policy'],
            {'kind': 'evaluate', 'source_job': job_id, 'checkpoint': source['checkpoint'], 'partition':partition})

    def benchmark(self, job_id):
        source=self.get(job_id)
        if source['status']!='completed' or not source['checkpoint'] or json.loads(source['config']).get('kind','train')!='train':
            raise ValueError('Выберите завершённое обучение')
        return self._insert(source['profile'],source['policy'],{'kind':'benchmark','source_job':job_id})

    def research(self):
        from .research import records
        if not any(r['outcome'] in {'success','failure'} for r in records()):
            raise ValueError('Нет диагностических эпизодов с рандомизированным назначением')
        return self._insert('intervention', 'sklearn', {'kind':'research'})

    def _insert(self, name, policy, cfg):
        ident = uuid.uuid4().hex
        with catalog.connect() as db:
            db.execute('INSERT INTO jobs(id,profile,policy,status,config,created) VALUES(?,?,?,?,?,?)',
                       (ident, name, policy, 'queued', json.dumps(cfg), time.time()))
        return ident

    def get(self, ident):
        entries = [j for j in catalog.rows('jobs') if j['id'] == ident]
        if len(entries) != 1:
            raise ValueError('Задание не найдено')
        return entries[0]

    def resume(self, ident):
        with self.lock:
            job = self.get(ident)
            if job['status'] not in {'failed', 'cancelled', 'interrupted'}:
                raise ValueError('Возобновляется только остановленное задание')
            cfg = json.loads(job['config'])
            if cfg.get('kind')=='sim_ppo':
                checkpoint=DATA/'jobs'/ident/'latest.zip'
                if not checkpoint.is_file():raise ValueError('Нет сохранённого PPO checkpoint')
                cfg['resume_checkpoint']=str(checkpoint)
                with catalog.connect() as db:
                    db.execute("UPDATE jobs SET config=?,status='queued',error=NULL WHERE id=?",(json.dumps(cfg),ident))
                return
            if cfg.get('kind', 'train') != 'train':
                raise ValueError('Для диагностики создайте новое задание')
            cfg['resume_checkpoint'] = str(resumable_checkpoint(ident))
            with catalog.connect() as db:
                db.execute("UPDATE jobs SET config=?,status='queued',error=NULL WHERE id=?", (json.dumps(cfg), ident))

    def tick(self):
        with self.lock:
            if (self.thread and self.thread.is_alive()) or self.control.owner is not None or self.control.maintenance:
                return
            queued = [j for j in reversed(catalog.rows('jobs')) if j['status'] == 'queued']
            if queued:
                self.cancel_requested.clear()
                self.active_id = queued[0]['id']
                self.thread = threading.Thread(target=self._run, args=(queued[0],), daemon=True)
                self.thread.start()

    def _run(self, job):
        ident = job['id']
        try:
            cfg=json.loads(job['config'])
            gpu=cfg.get('kind','train') in {'train','evaluate','benchmark'} and cfg.get('device','cuda').startswith('cuda')
            with ResourceLease('compute'), (ExistingTrainingLease() if gpu else contextlib.nullcontext()):
                path = DATA/'jobs'/f'{ident}.json'
                path.write_text(job['config'])
                with (DATA/'logs'/f'{ident}.log').open('a') as log:
                    with self.lock:
                        if self.cancel_requested.is_set():
                            return
                        with catalog.connect() as db:
                            db.execute("UPDATE jobs SET status='running' WHERE id=?", (ident,))
                        self.process = subprocess.Popen([sys.executable, '-m', 'explorer3.training', '--job', str(path)],
                            cwd=ROOT, env=offline_environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    code = self.process.wait()
                result_path = DATA/'jobs'/ident/'result.json'
                result = json.loads(result_path.read_text()) if code == 0 and result_path.exists() else None
                cancelled = self.cancel_requested.is_set()
                status = 'cancelled' if cancelled else 'completed' if result is not None else 'failed'
                with catalog.connect() as db:
                    db.execute('UPDATE jobs SET status=?,error=?,checkpoint=? WHERE id=?',
                        (status, None if result else f'Process exit={code}; см. журнал',
                         result.get('checkpoint') if result else None, ident))
        except ResourceBusy as exc:
            with catalog.connect() as db:
                db.execute("UPDATE jobs SET error=? WHERE id=? AND status='queued'",(str(exc),ident))
        except Exception as exc:
            catalog.event('error', f'Job {ident}: {exc}')
            with catalog.connect() as db:
                db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?", (str(exc), ident))
        finally:
            with self.lock:
                self.process = None
                self.active_id = None

    def cancel(self, ident=None):
        with self.lock:
            ident = ident or self.active_id
            if ident is None:
                return
            if ident == self.active_id:
                self.cancel_requested.set()
                if self.process and self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass  # Process exited between poll and kill.
            with catalog.connect() as db:
                db.execute("UPDATE jobs SET status='cancelled' WHERE id=? AND status IN ('queued','running')", (ident,))

    def log(self, ident):
        self.get(ident)
        path = DATA/'logs'/f'{ident}.log'
        text = path.read_text(errors='replace') if path.exists() else 'Задание ещё не запущено'
        for child in sorted((DATA/'jobs'/ident).glob('trainer-*.log')):
            text += '\n' + child.name + '\n' + child.read_text(errors='replace')[-16000:]
        return text[-24000:]
