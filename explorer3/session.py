"""Explicit physical session, bounded by the browser lease. No automatic reconnect."""
import threading
import time
import uuid
from . import catalog
from .hardware import HardwareSource
from .recording import EpisodeWriter
from .resources import ResourceLease

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

    def start(self, owner: str, name: str, scene: str, record=True):
        if name!='so101_pick_place':
            raise RuntimeError('Драйвер профиля ещё не принят')
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Сессия уже работает')
        if not scene.strip():
            raise ValueError('Нужно обозначение расстановки')
        session_id=self.control.acquire(owner)
        self.outcome='aborted'
        self.error=None
        self.frame_count=0
        self.latest=None
        self.scene=scene
        self.record_requested=record
        self.phase='preparing'
        self.thread=threading.Thread(target=self._run,args=(name,session_id,scene),daemon=True)
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
        lease=ResourceLease('compute')
        try:
            # A rollout/recording and trainer cannot own compute concurrently.
            lease.__enter__()
            if self.record_requested:
                writer=EpisodeWriter(name,session_id,scene)
            source=HardwareSource()
            provenance=source.start(explicit_user_start=True,stop=self.control.expired)
            source.begin_episode()
            self.phase='aligning'
            deadline=time.monotonic()
            while not self.control.expired():
                sample=source.sample(self.frame_count,stop=self.control.expired)
                with self.lock:
                    self.latest=sample
                self.phase=sample['teleoperation_phase']
                if writer:
                    writer.add(sample)
                self.frame_count+=1
                deadline+=1/30
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
            if source:
                errors=source.close()
                if errors:
                    catalog.event('error','; '.join(errors))
                    self.outcome='aborted'
            if writer:
                self.phase='saving'
                try:
                    provenance['stop_reason']=self.control.last_stop_reason
                    provenance['heartbeat_count']=self.control.heartbeat_count
                    writer.finish(self.outcome,provenance)
                except Exception as exc:
                    self.error=f'Finalize: {exc}'
                    catalog.event('error',self.error)
            lease.__exit__(None,None,None)
            if not self.error:
                self.phase='idle'
