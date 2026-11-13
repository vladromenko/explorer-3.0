"""Session authorization and expiring browser lease. Startup is always disarmed."""
import threading
import time
import uuid
from . import catalog
from .paths import DATA
from . import local_work
from agents.tank.motion_permission import bench_authorized, BENCH_V, BENCH_W

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
        self.base_keys=set()
        self.base_inputs={}
        self.maintenance=False

    def authorized(self):
        if self.robot=='tank' and bench_authorized(DATA/'configs'):
            return True
        if not local_work.hardware_allowed(self.robot):
            return False
        if self.robot=='so101' and (DATA/'configs/so101-external-owner.json').exists():
            return False
        if self.robot=='tank' and (DATA/'validation/tank-motion-incident.json').exists():
            return False
        # Installed only after the operator authorizes the concrete physical block.
        stage='b' if self.robot=='so101' else 'c' if self.robot=='tank' else 'd'
        return (DATA/f'stage-{stage}-authorized.json').is_file()

    def select(self, robot: str):
        if robot not in {'so101','tank','rosmaster'}:
            raise ValueError('Unknown robot')
        if self.maintenance:
            raise RuntimeError('Дождитесь передачи управления')
        self.stop('Смена робота: удержанные команды отменены')
        self.robot=robot

    def acquire(self, owner: str):
        with self.lock:
            if self.maintenance:
                raise RuntimeError('Идёт передача управления')
            if not self.authorized():
                raise PermissionError('Для выбранного робота ещё не разрешён физический блок')
            if self.owner is not None:
                raise RuntimeError('Ручное управление уже занято')
            if self.robot not in {'so101','tank'}:
                raise RuntimeError('Драйвер этого робота ещё не проверен')
            self.base_keys.clear()
            self.base_inputs.clear()
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
            self.base_keys.clear()
            self.base_inputs.clear()
            self.stop_event.set()
            self.owner=None
            self.expires=0
            self.mode='Наблюдение'
        catalog.event('info',reason)

    def watchdog(self):
        if self.owner is not None and self.robot=='tank' and not self.authorized():
            self.stop('Разрешение проверки Tank истекло')
        if self.owner is not None and self.expired():
            self.stop('Lease истёк: связь с оператором потеряна')

    def base_key(self,owner,key,pressed,source=None):
        with self.lock:
            if key not in {'up','down','left','right','up_left','up_right','down_left','down_right'}:
                raise ValueError('Unknown driving key')
            source=source or key
            if not isinstance(source,str) or len(source)>64:raise ValueError('Invalid input source')
            if self.robot=='tank' and self.owner==owner and not self.expired():
                if pressed:self.base_inputs[source]=key
                else:self.base_inputs.pop(source,None)
                self.base_keys=set(self.base_inputs.values())

    def base_action(self):
        with self.lock:
            if self.robot!='tank' or self.expired():return [0.,0.]
            keys={part for key in self.base_keys for part in key.split('_')}
            forward=int('up' in keys)-int('down' in keys)
            turn=int('left' in keys)-int('right' in keys)
            # Reverse diagonals describe the path of the rear of the chassis:
            # down-left backs into a left arc, using the same native v/omega channels.
            return [BENCH_V*forward,BENCH_W*turn*(-1 if forward<0 else 1)]
