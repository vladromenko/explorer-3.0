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

    def enqueue(self, name, policy, smoke=True, sim_model_hash=None):
        profile(name)
        if policy not in {'act', 'smolvla','diffusion'}:
            raise ValueError('Unknown trainer')
        from .policy_catalog import validate_recipe
        validate_recipe(policy,name)
        plan_id, _ = freeze(name, smoke,sim_model_hash=sim_model_hash)
        cfg={'kind': 'train', 'profile': name, 'policy': policy,
            'smoke': bool(smoke), 'plan_id': plan_id, 'seed': 1000}
        if profile(name).get('domain')=='sim' and not smoke:
            recipe=json.loads((ROOT/'configs/sim-budgets.json').read_text())[name][policy]
            cfg.update(steps=recipe['steps'],batch_size=recipe['batch_size'],device=recipe['device'],seed=recipe['seed'])
        return self._insert(name,policy,cfg)

    def cognitive_simulation(self,comparison='M3',seed=17,scene='workshop'):
        from .sim.room import PRESETS
        if scene not in PRESETS:raise ValueError('Unknown scene')
        if comparison not in {'M0','M1','M2','M3'}:raise ValueError('Unknown comparison')
        if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid scene seed')
        if any(j['status'] in {'queued','running'} and json.loads(j['config']).get('kind')=='cognitive_simulation' for j in catalog.rows('jobs')):
            raise ValueError('Проверка команды уже выполняется; дождитесь результата или нажмите STOP')
        return self._insert('fleet_lab_sim','cognitive',{'kind':'cognitive_simulation','comparison':comparison,
            'seed':seed,'scene':scene,'device':'cuda','training_requested':False,'real_authorized':False})

    def fleet_execute(self,selected,**options):
        from .fleet_runs import prepare
        config=prepare(selected,**options)
        return self._insert('fleet_lab_sim','frozen_fleet',config)

    def train_prepared_gain(self,ident):
        from .gain_preparation import load
        load(ident)
        return self._insert('intervention','branch_gain',{'kind':'branch_gain_train','plan_id':ident,'device':'cpu'})

    def identify_rosmaster(self,ident):
        from .rosmaster_preparation import load_identification
        plan=load_identification(ident)
        return self._insert(plan['profile'],'sysid',{'kind':'rosmaster_identification','plan_id':ident,'device':'cpu'})

    def start_prepared_training(self,draft_id):
        from .training_drafts import load
        draft=load(draft_id)
        config={**draft,'draft_id':draft_id,'preparation_only':False,'training_requested':True}
        config.pop('training_started',None)
        # One active execution per immutable draft, including requests from two
        # browser tabs/processes. Completed experiments may be repeated explicitly.
        with catalog.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("""SELECT id FROM jobs WHERE status IN ('queued','running')
                AND json_extract(config,'$.draft_id')=?""",(draft_id,)).fetchone()
            if row:return row['id']
            ident=uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,profile,policy,status,config,created) VALUES(?,?,?,?,?,?)',
                (ident,draft['profile'],draft['policy'],'queued',json.dumps(config),time.time()))
        return ident

    def sim_ppo(self, name, sim_model_hash=None,variation='nominal'):
        if name not in {'so101_pick_place_sim','rosmaster_pick_place_sim','rosmaster_dock_sim','tank_dock_sim'}:
            raise ValueError('Среда ещё не принята')
        if variation not in {'nominal','varied'}:raise ValueError('Unknown scene conditions')
        budget=json.loads((ROOT/'configs/sim-budgets.json').read_text())[name]['pick_place_ppo']
        return self._insert(name,'ppo',{'kind':'sim_ppo','profile':name,'steps':budget['timesteps'],
            'seed':budget['seed'],'device':'cpu','domain':'sim','expected_model_hash':sim_model_hash,'variation':variation})

    def sim_evaluate(self, job_id):
        source=self.get(job_id)
        if source['status']!='completed' or source['policy'] not in {'act','smolvla','diffusion'} or not source['checkpoint']:
            raise ValueError('Выберите завершённое обучение политики')
        if profile(source['profile']).get('domain')!='sim':
            raise ValueError('Нужен симуляционный профиль')
        from .datasets import load_plan,select_simulation_revision
        config=json.loads(source['config'])
        if config.get('plan_id'):
            split=load_plan(config['plan_id'])['splits']
        else:
            split=json.loads((DATA/'jobs'/job_id/'split.json').read_text())
        _,model_hash=select_simulation_revision([e for group in split.values() for e in group])
        if model_hash is None:raise ValueError('Нет версии физики обучающего набора')
        return self._insert(source['profile'],source['policy'],{'kind':'sim_evaluate','profile':source['profile'],
            'source_job':job_id,'checkpoint':source['checkpoint'],'device':'cpu',
            'expected_model_hash':model_hash,
            'episodes':3 if json.loads(source['config']).get('smoke') else 20})

    def compare_sim_branches(self,name,*,source_job=None,seed=14,warmup_s=.5,delay_s=.5,correction_s=5.,horizon_s=15.):
        from .branching_config import validate_config
        if profile(name).get('domain')!='sim':raise ValueError('Только симуляция')
        cfg={'kind':'sim_branches','profile':name,'seed':int(seed),'warmup_s':float(warmup_s),
            'delay_s':float(delay_s),'correction_s':float(correction_s),'horizon_s':float(horizon_s),
            'repeats':2,'device':'cpu','capture_correction':True,'render_preview':True}
        validate_config(cfg)
        policy='diagnostic_hold'
        if source_job:
            source=self.get(source_job)
            if source['profile']!=name or source['status']!='completed' or not source['checkpoint']:
                raise ValueError('Нужен завершённый checkpoint выбранного профиля')
            if source['policy'] not in {'act','smolvla','diffusion'}:raise ValueError('Неподдерживаемая политика')
            from .datasets import load_plan,select_simulation_revision
            plan=json.loads(source['config']).get('plan_id')
            if not plan:raise ValueError('Нет зафиксированного плана обучающих данных')
            groups=load_plan(plan)['splits']
            _,model_hash=select_simulation_revision([e for group in groups.values() for e in group])
            if model_hash is None:raise ValueError('Нет версии модели обучающих данных')
            cfg.update(source_job=source_job,checkpoint=source['checkpoint'],expected_model_hash=model_hash)
            policy=source['policy']
        return self._insert(name,policy,cfg)

    def accept_fleet_correction(self,source_job,index,reviewed=False):
        if reviewed is not True:raise ValueError('Сначала проверьте сегмент и исход задачи')
        from .fleet_corrections import validate
        from .branch_corrections import source_hash
        source=self.get(source_job)
        if source['status']!='completed' or json.loads(source['config']).get('kind')!='fleet_execution':
            raise ValueError('Нужен завершённый общий прогон')
        result=json.loads((DATA/'jobs'/source_job/'result.json').read_text())
        entries=result.get('correction_segments',[])
        if type(index) is not int or not 0<=index<len(entries):raise ValueError('Нет такого сегмента')
        root=Path(entries[index]['spool']);meta=validate(root)
        if meta['lineage']!=result['lineage']:raise ValueError('Сегмент из другого прогона')
        review={'decision':'accept_fleet_sim_correction','source_sha256':source_hash(root),'lineage':meta['lineage'],
            'source_job':source_job,'recorded_wall_unix_s':time.time(),'source':'scripted_expert','human_demonstration':False}
        return self._insert(meta['profile'],'dataset',{'kind':'sim_correction_import','profile':meta['profile'],
            'spool':str(root),'review':review,'device':'cpu'})

    def accept_sim_correction(self,source_job,reviewed=False):
        if reviewed is not True:raise ValueError('Сначала проверьте кадры и результат исправления')
        from .branch_corrections import candidate,source_hash
        source=self.get(source_job)
        if source['status']!='completed' or json.loads(source['config']).get('kind')!='sim_branches':
            raise ValueError('Нужен завершённый отчёт сравнения')
        report=json.loads((DATA/'jobs'/source_job/'result.json').read_text())
        root,meta=candidate(report)
        review={'decision':'accept_sim_scripted_correction','source_sha256':source_hash(root),
            'source_job':source_job,'lineage':report['lineage'],'snapshot_sha256':report['snapshot_sha256'],
            'recorded_wall_unix_s':time.time(),'source':'scripted_expert','human_demonstration':False}
        return self._insert(source['profile'],'dataset',{'kind':'sim_correction_import',
            'profile':source['profile'],'spool':str(root),'review':review,'device':'cpu'})

    def inspect_image(self,image,method,prompt,source):
        import base64,io
        from PIL import Image
        if method not in {'yoloe','sam3'} or not isinstance(prompt,str) or not 1<=len(prompt.strip())<=200:
            raise ValueError('Выберите модель и короткое название объекта')
        raw=base64.b64decode(image,validate=True)
        if len(raw)>8_000_000:raise ValueError('Изображение слишком большое')
        frame=Image.open(io.BytesIO(raw))
        if frame.width*frame.height>4096*4096:raise ValueError('Изображение слишком большое')
        frame.load()
        root=DATA/'vision'/uuid.uuid4().hex;root.mkdir(parents=True)
        frame.convert('RGB').save(root/'input.png')
        (root/'source.json').write_text(json.dumps({'source':source,'captured_monotonic_ns':time.monotonic_ns(),
            'domain':'sim' if source in {'individual_sim','fleet_sim'} else 'uploaded','ground_truth':False}))
        return self._insert('vision',method,{'kind':'vision','method':method,'image':str(root/'input.png'),
            'prompt':prompt.strip(),'device':'cuda' if method=='sam3' else 'cpu'})

    def prepare_support(self,job_id):
        from .act_support import source_plan
        source=self.get(job_id);source_plan(source)
        return self._insert(source['profile'],'act',{'kind':'support_calibration','source_job':job_id,
            'profile':source['profile'],'device':'cpu'})

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
                from .sim.checkpoints import latest
                try:checkpoint=latest(DATA/'jobs'/ident)
                except (OSError,ValueError) as exc:raise ValueError('Нет согласованной пары PPO и нормировки для продолжения') from exc
                cfg['resume_checkpoint']=str(checkpoint)
                with catalog.connect() as db:
                    db.execute("UPDATE jobs SET config=?,status='queued',error=NULL WHERE id=?",(json.dumps(cfg),ident))
                return
            if cfg.get('kind', 'train') != 'train':
                raise ValueError('Для диагностики создайте новое задание')
            cfg['resume_checkpoint'] = str(resumable_checkpoint(ident))
            cfg['resume_of_existing_job']=ident
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
            gpu=cfg.get('kind','train') in {'train','evaluate','benchmark','vision','cognitive_simulation'} and cfg.get('device','cuda').startswith('cuda')
            with ResourceLease('compute'), (ExistingTrainingLease() if gpu else contextlib.nullcontext()):
                path = DATA/'jobs'/f'{ident}.json'
                path.write_text(job['config'])
                with (DATA/'logs'/f'{ident}.log').open('a') as log:
                    with self.lock:
                        if self.cancel_requested.is_set():
                            return
                        with catalog.connect() as db:
                            db.execute("UPDATE jobs SET status='running',error=NULL WHERE id=?", (ident,))
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
            mission_file=DATA/'jobs'/ident/'mission.json'
            if mission_file.is_file():
                from .cognitive import broker
                broker.cancel(json.loads(mission_file.read_text())['mission_id'],'Job cancelled')
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
