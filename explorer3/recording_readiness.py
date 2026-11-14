"""Read-only recording preflight. Never acquires a robot or invokes a model."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time
import httpx
from . import fleet_config, catalog
from .paths import DATA,ROOT

NAMES={'so101':'SO-101','tank':'Tank','rosmaster':'ROSMASTER'}


def item(key,label,ok,value,action):
    return {'key':key,'label':label,'status':'green' if ok else 'red',
            'value':value,'action':'' if ok else action}


def finite(value):return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def clock_check(before_ns,after_ns,remote_ns):
    if not finite(remote_ns) or remote_ns<=0:return {'offset_ms':None,'uncertainty_ms':None,'ok':False}
    offset=((before_ns+after_ns)/2-remote_ns)/1e6
    uncertainty=(after_ns-before_ns)/2e6
    return {'offset_ms':offset,'uncertainty_ms':uncertainty,
            'ok':abs(offset)+uncertainty<=20}


def evaluate(robot,evidence,allowed,reason):
    h=evidence.get('health') or {};sensor=evidence.get('sensors') or {};machine=evidence.get('machine') or {}
    now=time.time();checks=[item('network','Связь',bool(h) or machine.get('reachable',False),
        'Есть ответ' if h or machine.get('reachable') else 'Нет ответа','Включите робот и проверьте подключение к Wi-Fi')]
    checks.append(item('agent','Агент Explorer',bool(h),'Работает' if h else 'Не запущен',
        'Откройте управление и нажмите «Включить управление», когда готовы к физической проверке'))
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'agents'/robot).glob('*.py')}
    hashes=h.get('software') or machine.get('software') or {}
    required={'agent.py','runtime.py','leased_core.py','control_adapter.py'} if robot=='rosmaster' else {'agent.py','native_driver.py','native_client.py','limits.py','motor_link.py','motion_permission.py','sensors.py'}
    version_ok=required<=hashes.keys() and all(expected.get(k)==hashes[k] for k in required)
    checks.append(item('version','Версия агента',version_ok,
        'Совпадает' if version_ok else 'Не подтверждена',
        'Обновите собственный агент Explorer из текущей версии проекта'))
    checks.append(item('permission','Допуск пользователя',allowed,reason,'Включите допуск этого робота ниже'))
    status=h.get('status') or h.get('observation') or h
    voltage=status.get('battery_voltage');threshold=h.get('battery_stop_v',6.9 if robot=='tank' else None)
    stamp=status.get('battery_received_ns');age=status.get('battery_age_s')
    if age is None and stamp and h.get('monotonic_ns'):age=(h['monotonic_ns']-stamp)/1e9
    if not finite(voltage):
        battery=sensor.get('streams',{}).get('battery',{})
        voltage=battery.get('voltage');age=battery.get('age_s')
    fresh=finite(age) and 0<=age<=2
    checks.append(item('battery','Батарея',finite(voltage) and finite(threshold) and voltage>=threshold and fresh,
        {'voltage':voltage,'stop_below_v':threshold,'age_s':age},
        'Откройте управление и получите свежее показание батареи' if not fresh else 'Зарядите выключенный робот перед записью'))
    for stream,label in (('rgb','Камера'),('depth','Глубина'),('scan','Лидар'),('imu','IMU')):
        row=sensor.get('streams',{}).get(stream,{})
        age=row.get('age_s');hz=row.get('rate_hz')
        ok=finite(age) and 0<=age<=.5 and finite(hz) and hz>0
        if stream=='rgb':ok=ok and hz>=10
        checks.append(item(stream,label,ok,{'age_s':age,'hz':hz},
            'Проверьте подключение '+label.lower()+' и нагрузку компьютера робота'))
    rtt=evidence.get('rtt_p95_ms')
    checks.append(item('latency','Задержка связи',finite(rtt) and rtt<=150,rtt,'Поднесите робот ближе к точке Wi-Fi'))
    clock=evidence.get('clock') or {}
    checks.append(item('clock','Согласование времени',clock.get('ok',False),clock,'Синхронизируйте часы Homebrain и робота; повторите проверку сети'))
    fault=h.get('fault') or status.get('error') or h.get('guard_reason') or status.get('base_guard') or status.get('guard') or status.get('arm_blocked')
    checks.append(item('guard','Последняя защита',not fault,{'reason':fault,'observed_at':now},
        'Устраните указанную причину и заново включите управление; после USB-отказа подтвердите остановку'))
    if robot=='tank':
        link=h.get('motor_link') or {}
        checks.append(item('usb','USB за сегодня',not link.get('pending',False),link,
            'Проверьте кабель моторной платы; подтвердите остановку кнопкой «Продолжить»'))
        checks.append(item('computer','Компьютер Tank',machine.get('throttled')==0,
            {k:machine.get(k) for k in ('temperature_c','throttled','cpu_percent','sampled_at')},
            'Проверьте питание и охлаждение Tank'))
    else:
        measured=status.get('arm_measured');ages=status.get('arm_measured_ages_s',status.get('arm_feedback_age_s'))
        arm_ok=isinstance(measured,list) and len(measured)==6 and all(finite(v) for v in measured)
        if ages is not None:arm_ok=arm_ok and all(finite(v) and v<=.5 for v in ages)
        else:arm_ok=arm_ok and status.get('arm_complete',False)
        checks.append(item('arm','Обратная связь руки',arm_ok,measured,'Проверьте питание руки и связь с контроллером'))
    return {'robot':robot,'name':NAMES[robot],'checks':checks,'ready':all(c['status']=='green' for c in checks)}


# Fixed read-only script; no paths, tokens or command arguments from a browser.
MACHINE="""import json,time,subprocess,hashlib
from pathlib import Path
root=Path.home()/'explorer3'
t=Path('/sys/class/thermal/thermal_zone0/temp')
stat=lambda:list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
a=stat();time.sleep(.1);b=stat();total=sum(b)-sum(a);idle=b[3]+b[4]-a[3]-a[4]
try:
 p=subprocess.run(['vcgencmd','get_throttled'],capture_output=True,text=True,timeout=2)
 throttle=int(p.stdout.strip().split('=')[-1],16) if p.returncode==0 else None
except OSError:throttle=None
print(json.dumps({'reachable':True,'temperature_c':float(t.read_text())/1000 if t.exists() else None,
 'throttled':throttle,'cpu_percent':100*(1-idle/total) if total else None,'software':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('agent.py','runtime.py','leased_core.py','control_adapter.py') if (root/n).is_file()},
 'sampled_at':time.time()}))
"""


class Service:
    def __init__(self,local_state=lambda:{}):
        self.local_state=local_state;self.lock=threading.Lock();self.last=None;self.at=0
        self.machine_cache={};self.rtts={name:deque(maxlen=30) for name in ('tank','rosmaster')}
        self.rate_samples={}

    def machine(self,robot,cfg):
        cached=self.machine_cache.get(robot)
        if cached and time.monotonic()-cached[0]<30:return cached[1]
        ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=2']
        if robot=='tank':ssh+=['tank']
        else:ssh+=['-i',str(Path.home()/'.ssh/rosmaster-collector'),str(cfg.get('ssh_user','vlad'))+'@'+cfg['host']]
        try:
            result=subprocess.run(ssh+[shlex.join(['python3','-c',MACHINE])],capture_output=True,text=True,timeout=5,check=True)
            value=json.loads(result.stdout)
        except (OSError,ValueError,subprocess.SubprocessError):value={'reachable':False,'sampled_at':time.time()}
        self.machine_cache[robot]=(time.monotonic(),value)
        return value

    def remote(self,robot):
        evidence={}
        try:cfg=json.loads((DATA/'configs'/f'{robot}-agent.json').read_text())
        except (OSError,ValueError):return evidence
        evidence['machine']=self.machine(robot,cfg)
        with httpx.Client(headers={'X-Explorer-Token':cfg['token']},timeout=.65,trust_env=False) as client:
            try:
                before=time.monotonic();r=client.get(f"http://{cfg['host']}:{cfg['port']}/health");r.raise_for_status()
                evidence['health']=r.json();self.rtts[robot].append((time.monotonic()-before)*1000)
                evidence['rtt_p95_ms']=sorted(self.rtts[robot])[max(0,math.ceil(len(self.rtts[robot])*.95)-1)]
            except (httpx.HTTPError,ValueError):pass
            candidates=[]
            for port in ((8043,8044) if robot=='tank' else (8053,8054)):
                try:
                    before=time.time_ns();r=client.get(f"http://{cfg['host']}:{port}/health");after=time.time_ns();r.raise_for_status()
                    value=r.json();value['_clock']=clock_check(before,after,(value.get('clock_pair') or {}).get('system_time_ns'))
                    candidates.append(value)
                except (httpx.HTTPError,ValueError):pass
            if candidates:
                best=max(candidates,key=lambda x:sum(v.get('age_s',999)<.5 for v in x.get('streams',{}).values()))
                evidence['clock']=best.pop('_clock');evidence['sensors']=best
                for key,row in best.get('streams',{}).items():
                    # Older observers expose received counters; estimate Hz without pretending it is measured in one sample.
                    stamp=time.monotonic();identity=(robot,key,best.get('boot_id'))
                    count=row.get('frame_count');previous=self.rate_samples.get(identity)
                    if row.get('rate_hz') is None and previous and finite(count) and count>=previous[1]:
                        row['rate_hz']=(count-previous[1])/max(.001,stamp-previous[0])
                    if finite(count):self.rate_samples[identity]=(stamp,count)
        return evidence

    def snapshot(self):
        with self.lock:
            if self.last and time.monotonic()-self.at<1.8:return self.last
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures={r:pool.submit(self.remote,r) for r in ('tank','rosmaster')}
                robots={r:evaluate(r,f.result(),*fleet_config.robot_allowed(r,DATA)) for r,f in futures.items()}
            local=self.local_state();allowed,reason=fleet_config.robot_allowed('so101',DATA)
            try:owner=fleet_config.load(DATA)['robots']['so101']['owner']
            except ValueError:owner='unknown'
            from .hardware import calibration_snapshot
            try:calibration_snapshot();calibrated=True
            except (OSError,ValueError,KeyError):calibrated=False
            checks=[item('permission','Владелец SO-101',allowed,owner+' · '+reason,'Откройте SO-101 и нажмите «Взять SO-101 в Explorer»'),
                item('leader','Leader',Path('/dev/so101-leader').exists(),'USB обнаружен' if Path('/dev/so101-leader').exists() else 'USB не найден','Подключите Leader к Homebrain'),
                item('follower','Follower',Path('/dev/so101-follower').exists(),'USB обнаружен' if Path('/dev/so101-follower').exists() else 'USB не найден','Подключите Follower к Homebrain'),
                item('camera','Камера SO-101',local.get('camera_age_s',999)<.5,{'age_s':local.get('camera_age_s'),'frames':local.get('frames')},'Откройте управление SO-101 и проверьте поступление кадров'),
                item('calibration','Калибровка',calibrated,'Проверена по хешам' if calibrated else 'Не найдена или изменена','Восстановите собственную копию проверенной калибровки Explorer')]
            robots['so101']={'robot':'so101','name':'SO-101','checks':checks,'ready':all(c['status']=='green' for c in checks)}
            free=shutil.disk_usage(DATA).free/1024**3
            jobs=[{'id':j['id'],'status':j['status']} for j in catalog.rows('jobs') if j['status'] in ('running','queued')]
            from .data_safety import status as backup_status
            backup=backup_status()
            self.last={'updated_at':time.time(),'robots':robots,'recording_disk_allowed':free>=8,
                'general':[item('disk','Свободное место',free>=15,round(free,2),'Перенесите сохранённые записи на внешний диск; ниже 8 ГБ новая запись запрещена'),
                    item('jobs','Активные задания',not jobs,jobs,'Дождитесь завершения задания перед записью'),
                    item('backup','Резервная копия',backup.get('available',False) and not backup.get('warning'),backup,
                         'Подключите внешний диск с папкой explorer3-backup и нажмите «Сделать копию сейчас»')],
                'local_activity':local,'physical_pass':False}
            self.at=time.monotonic();return self.last
