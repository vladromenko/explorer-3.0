"""Explicit physical session, bounded by the browser lease. No automatic reconnect."""
import threading
import time
import uuid
from contextlib import nullcontext
from . import catalog
from .hardware import HardwareSource
from .tank import TankSource
from .profiles import profile
from .recording import EpisodeWriter
from .resources import ResourceLease
from agents.tank.motion_permission import bench_authorized
from .paths import DATA

class Session:
    def __init__(self, control):
        self.control=control
        self.thread=None
        self.lock=threading.RLock()
        self.outcome='aborted'
        self.latest=None
        self.error=None
        self.frame_count=0
        self.scene='initial'
        self.phase='idle'
        self.record_requested=False
        self.trial_plan=None
        self.category="demonstration"

    def start(self, owner: str, name: str, scene: str, record=True, collection_id=None, category="demonstration"):
        if name=='tank_dock' and record and bench_authorized(DATA/'configs'):
            raise PermissionError('Проверка на подставке: запись показов пока запрещена')
        if name not in {'so101_pick_place','tank_dock'}:
            raise RuntimeError('Драйвер профиля ещё не принят')
        if profile(name)['robot']!=self.control.robot:
            raise ValueError('Профиль не соответствует выбранному роботу')
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Сессия уже работает')
        if not scene.strip():
            raise ValueError('Нужно обозначение расстановки')
        if category not in {'demonstration','correction'}:raise ValueError('Unknown human segment category')
        self.category=category
        self.trial_plan=None
        session_id=self.control.acquire(owner)
        self.outcome='aborted'
        self.error=None
        self.frame_count=0
        self.latest=None
        self.scene=scene
        self.record_requested=record
        self.phase='preparing'
        self.thread=threading.Thread(target=self._run,args=(name,collection_id or session_id,scene),daemon=True)
        self.thread.start()

    def start_trial(self,owner,trial_id,scene):
        from . import trials
        plan=trials.load(trial_id)
        trials.require_physical(plan)
        if profile(plan['profile'])['robot']!=self.control.robot:
            raise ValueError('Выберите робота плана испытания')
        if self.thread and self.thread.is_alive():raise RuntimeError('Сначала остановите текущую сессию')
        session_id=self.control.acquire(owner)
        self.trial_plan=plan
        self.outcome='aborted';self.error=None;self.frame_count=0;self.latest=None
        self.scene=scene;self.record_requested=True;self.phase='preparing'
        self.thread=threading.Thread(target=self._run,args=(plan['profile'],session_id,scene),daemon=True)
        self.thread.start()

    def finish(self, outcome: str, reason: str | None = None):
        if outcome not in {'success','failure','aborted'}:
            raise ValueError('Invalid outcome')
        self.outcome=outcome
        self.control.stop(reason or ('Эпизод закончен: '+outcome))

    def _run(self,name,session_id,scene):
        source=None
        writer=None
        provenance={}
        provider=None
        # Manual control uses no GPU; it must not block the other robot's session.
        lease=ResourceLease('compute') if self.record_requested or self.trial_plan else nullcontext()
        try:
            # A rollout/recording and trainer cannot own compute concurrently.
            lease.__enter__()
            if self.record_requested:
                writer=EpisodeWriter(name,session_id,scene,category='rollout' if self.trial_plan else self.category)
            if self.trial_plan:
                from .trials import require_physical,ReplayActions,PolicyActions
                require_physical(self.trial_plan)
                provider=(ReplayActions if self.trial_plan['kind']=='replay' else PolicyActions)(self.trial_plan)
            if name=='tank_dock':
                if self.control.expired():raise InterruptedError('STOP before Tank handoff')
                from .tank_handoff import invoke
                invoke('enter')
                if self.control.expired():raise InterruptedError('STOP during Tank handoff')
            source=(HardwareSource(teleoperate=not bool(provider)) if name=='so101_pick_place'
                    else TankSource(self.control,mode=self.trial_plan['kind'] if self.trial_plan else 'manual'))
            provenance=source.start(explicit_user_start=True,stop=self.control.expired)
            source.begin_episode()
            self.phase='aligning'
            deadline=time.monotonic()
            trial_start=deadline
            if provider and self.trial_plan['kind']=='policy':
                # Warm-up is discarded; observe again and reset chunks before any policy command.
                observation=source.observe()
                if name=='tank_dock':source.plan_action(provider,observation)
                else:provider.next(observation)
                provider.reset()
                deadline=time.monotonic();trial_start=deadline
            while not self.control.expired():
                if provider:
                    if time.monotonic()-trial_start>self.trial_plan['max_duration_s']:
                        self.control.stop('Лимит времени испытания');break
                    observation=source.observe()
                    try:
                        requested=(source.plan_action(provider,observation)
                                   if name=='tank_dock' and self.trial_plan['kind']=='policy' else provider.next(observation))
                    except StopIteration:
                        self.control.stop('Повтор завершён без отметки исхода; запись сохранится как aborted');break
                    if self.control.expired():raise InterruptedError('STOP before trial command')
                    action=source.send(requested,observation)
                    sample={**observation,**action,'frame_id':self.frame_count,'command_source':'policy',
                            'controller_kind':self.trial_plan['kind'],'action_requested':requested,
                            'trial_id':self.trial_plan['id'],'teleoperation_phase':'trial',
                            **getattr(provider,'decision_info',{})}
                else:
                    sample=source.sample(self.frame_count,stop=self.control.expired)
                with self.lock:
                    self.latest=sample
                self.phase=sample['teleoperation_phase']
                if writer:
                    writer.add(sample)
                self.frame_count+=1
                deadline+=1/profile(name)['fps']
                if provider:
                    lag=time.monotonic()-deadline
                    if lag>.5:raise RuntimeError('Испытание не выдерживает частоту; продолжение запрещено')
                    if lag>0:deadline=time.monotonic()+1/profile(name)['fps']
                self.control.stop_event.wait(max(0,deadline-time.monotonic()))
        except InterruptedError as exc:
            if not self.control.stop_event.is_set():
                self.error=str(exc)
                self.outcome='aborted'
                catalog.event('warning',str(exc))
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.phase='error'
            self.outcome='aborted'
            catalog.event('error',self.error)
        finally:
            if self.control.last_stop_reason is None and self.control.expired():
                self.control.stop('Lease истёк: heartbeat отсутствовал более 1 секунды')
            self.control.stop('Физическая сессия закрывается; автоматического продолжения нет')
            if provider:
                try:provider.reset()
                except Exception as exc:catalog.event('error','Policy reset during close: '+str(exc))
            if source:
                errors=source.close()
                if errors:
                    catalog.event('error','; '.join(errors))
                    self.outcome='aborted'
            if writer:
                self.phase='saving'
                try:
                    if self.trial_plan:provenance['trial_plan']=self.trial_plan
                    provenance['stop_reason']=self.control.last_stop_reason
                    provenance['heartbeat_count']=self.control.heartbeat_count
                    writer.finish(self.outcome,provenance)
                except Exception as exc:
                    self.error=f'Finalize: {exc}'
                    catalog.event('error',self.error)
            lease.__exit__(None,None,None)
            if not self.error:
                self.phase='idle'
