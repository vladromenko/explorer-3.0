"""Session authorization and expiring browser lease. Startup is always disarmed."""
import threading
import time
import uuid
from . import catalog
from .paths import DATA

class Control:
    def __init__(self):
        self.robot='so101'
        self.mode='Наблюдение'
        self.owner=None
        self.expires=0.0
        self.stop_event=threading.Event()
        self.stop_event.set()
        self.lock=threading.RLock()
        self.session=None
        self.last_stop_reason=None
        self.last_heartbeat=None
        self.heartbeat_count=0

    def authorized(self):
        # Installed only after the operator authorizes the concrete physical block.
        return (DATA/'stage-b-authorized.json').is_file()

    def select(self, robot: str):
        if robot not in {'so101','tank','rosmaster'}:
            raise ValueError('Unknown robot')
        self.stop('Смена робота: удержанные команды отменены')
        self.robot=robot

    def acquire(self, owner: str):
        with self.lock:
            if not self.authorized():
                raise PermissionError('Этап A: движение запрещено; требуется физическая приёмка B')
            if self.owner is not None:
                raise RuntimeError('Ручное управление уже занято')
            if self.robot != 'so101':
                raise RuntimeError('Драйвер этого робота ещё не проверен')
            self.last_stop_reason=None
            self.last_heartbeat=time.monotonic()
            self.heartbeat_count=0
            self.owner=owner
            self.session=uuid.uuid4().hex
            self.expires=time.monotonic()+1.0
            self.stop_event.clear()
            self.mode='Explorer 3.0'
            return self.session

    def heartbeat(self, owner: str):
        with self.lock:
            if self.owner == owner and not self.stop_event.is_set():
                self.last_heartbeat=time.monotonic()
                self.heartbeat_count+=1
                self.expires=self.last_heartbeat+1.0

    def expired(self):
        return self.stop_event.is_set() or time.monotonic()>self.expires

    def stop(self, reason='STOP'):
        with self.lock:
            if self.owner is not None or self.last_stop_reason is None:
                self.last_stop_reason=reason
            self.stop_event.set()
            self.owner=None
            self.expires=0
            self.mode='Наблюдение'
        catalog.event('info',reason)

    def watchdog(self):
        if self.owner is not None and self.expired():
            self.stop('Lease истёк: связь с оператором потеряна')
