"""Operator-performed acceptance with read-only evidence. Contains NO command client."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import threading
import time
import uuid
import httpx
from .paths import DATA
from .fleet_config import atomic_write,load

STEPS={
 'tank':[
  ('charger','Отсоедините зарядник. Включите управление без движения; наблюдайте напряжение 10 секунд.'),
  ('forward','Удерживайте ↑ 2 секунды, отпустите. Подтвердите движение вперёд и остановку.'),
  ('backward','Удерживайте ↓ 2 секунды, отпустите. Подтвердите движение назад и остановку.'),
  ('left','Удерживайте ← 2 секунды, отпустите. Подтвердите поворот влево и остановку.'),
  ('right','Удерживайте → 2 секунды, отпустите. Подтвердите поворот вправо и остановку.'),
  ('release','Начните короткое движение и отпустите кнопку. Гусеницы должны остановиться.'),
  ('stop','Начните короткое движение и нажмите STOP. Убедитесь, что гусеницы остановились.'),
  ('straight','На свободном полу проедьте вперёд по размеченному отрезку 1 м. Не поворачивайте; остановитесь.'),
  ('lidar','На малой скорости подъедьте к коробке в плоскости лидара. Должна сработать остановка до касания; будьте рядом с выключателем.'),
  ('hidden','Во время короткого движения переключитесь на другую вкладку на 2–3 секунды. После возвращения движение не должно возобновиться.'),
  ('sensors','Проверьте в камере сцену и в обзоре лидара коробку; глубина должна обновляться.'),
  ('recording','Создайте одну пробную запись в «Сборе датасета» и дождитесь её проверки.')],
 'rosmaster':[
  ('ready','Зарядник отключён, вокруг базы и руки свободно. Включите управление в панели.'),
  ('forward','Удерживайте W 2 секунды, отпустите; проверьте движение вперёд и остановку.'),
  ('backward','Удерживайте S 2 секунды, отпустите; проверьте движение назад и остановку.'),
  ('left','Удерживайте A 2 секунды, отпустите; проверьте движение влево и остановку.'),
  ('right','Удерживайте D 2 секунды, отпустите; проверьте движение вправо и остановку.'),
  ('release','Начните короткое движение базы и отпустите кнопку; проверьте остановку.'),
  ('base_stop','Во время короткого движения базы нажмите STOP и убедитесь, что колёса остановились.'),
  ('joints','Выберите «Суставы». По очереди немного измените каждый из пяти суставов в обе стороны, без предметов.'),
  ('gripper','Без предметов немного откройте и закройте захват; проверьте весь требуемый диапазон.'),
  ('stop','Во время небольшого движения руки нажмите STOP; убедитесь, что рука остановилась.'),
  ('hidden','Начните короткое движение базы, уйдите на другую вкладку на 2–3 секунды; проверьте остановку без возобновления.'),
  ('sensors','Проверьте камеру, глубину, оба лидара и IMU в обзоре датчиков.'),
  ('recording','Сохраните пробную запись руки или базы в «Сборе датасета» и дождитесь проверки.')],
 'so101':[
  ('ownership','Нажмите «Взять SO-101 в Explorer», подтвердите остановку :8020. Движение пока не включайте.'),
  ('leader','Включите телеперацию, дождитесь сближения; слегка подвигайте Leader по всем пяти суставам.'),
  ('gripper','Немного откройте и закройте захват Leader; проверьте повторение Follower.'),
  ('stop','Во время небольшого движения Leader нажмите STOP; проверьте остановку Follower.'),
  ('hidden','Включите телеперацию, слегка двигайте Leader и переключитесь на другую вкладку на 2–3 секунды. Проверьте остановку без возобновления.'),
  ('recording','Сохраните пробный перенос кубика в «Сборе датасета» и дождитесь проверки.')]
}


def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def vector(x):return isinstance(x,(list,tuple)) and bool(x) and all(finite(v) for v in x)


def normalize(robot,h,sensors=None):
    now=time.time();s=h.get('status') or h.get('observation') or h
    remote=h.get('monotonic_ns',0);streams=(sensors or {}).get('streams',{})
    camera=h.get('camera') or s.get('camera') or {};stamp=camera.get('received_ns')
    fresh=lambda name:streams.get(name,{}).get('age_s',999)<.5
    battery=s.get('battery_voltage');age=s.get('battery_age_s')
    if age is None and h.get('battery_received_ns'):age=(remote-h['battery_received_ns'])/1e9
    if not finite(battery):battery=streams.get('battery',{}).get('voltage');age=streams.get('battery',{}).get('age_s')
    imu=h.get('imu');yaw=None
    if robot=='tank' and vector(imu) and len(imu)>=6 and h.get('imu_received_ns') is not None and 0<=remote-h['imu_received_ns']<500_000_000:
        yaw=math.radians(imu[5]) # native SDK gyro is degrees/s; published ROS IMU is SI
    return {'at':now,'robot':robot,'armed':bool(h.get('armed')),'hardware_open':bool(h.get('hardware_open')),
        'command':h.get('action') if robot=='tank' else s.get('base_command'),
        'measured_velocity':s.get('measured_velocity'),'arm':s.get('arm_measured'),
        'battery_v':battery,'battery_fresh':finite(age) and 0<=age<=2,
        'camera_fresh':bool(stamp and 0<=remote-stamp<500_000_000) or fresh('rgb'),
        'camera_sequence':camera.get('sequence',streams.get('rgb',{}).get('frame_count')),
        'scan_fresh':fresh('scan'),'rear_scan_fresh':fresh('scan_rear'),'depth_fresh':fresh('depth'),'imu_fresh':fresh('imu'),
        'yaw_rate_rad_s':yaw,'front_distance_m':h.get('front_distance_m'),
        'guard':h.get('guard_reason') or s.get('guard'),'reason':h.get('reason') or s.get('reason'),
        'fault':h.get('fault') or s.get('error'),'physical_stop_confirmed_by_telemetry':False}


class Reader:
    def __init__(self,robot,local_reader=None):
        self.robot=robot;self.local_reader=local_reader;self.sensor_at=0;self.sensors={}
        self.client=None
        if robot!='so101':
            cfg=json.loads((DATA/'configs'/f'{robot}-agent.json').read_text())
            self.host=cfg['host'];self.port=cfg['port']
            self.client=httpx.Client(headers={'X-Explorer-Token':cfg['token']},timeout=.4,trust_env=False)
    def read(self):
        if self.robot=='so101':
            value=self.local_reader() if self.local_reader else {}
            return {'at':time.time(),'robot':'so101','owner':load(DATA)['robots']['so101']['owner'],**value}
        r=self.client.get(f'http://{self.host}:{self.port}/health');r.raise_for_status();health=r.json()
        if time.monotonic()-self.sensor_at>1:
            values=[]
            for port in ((8043,8044) if self.robot=='tank' else (8053,8054)):
                try:
                    r=self.client.get(f'http://{self.host}:{port}/health');r.raise_for_status();values.append(r.json())
                except (httpx.HTTPError,ValueError):pass
            if values:self.sensors=max(values,key=lambda x:sum(v.get('age_s',999)<.5 for v in x.get('streams',{}).values()))
            if values:self.sensor_at=time.monotonic()
        extra=max(0,time.monotonic()-self.sensor_at)
        sensor={**self.sensors,'streams':{k:{**v,'age_s':v.get('age_s',999)+extra} for k,v in self.sensors.get('streams',{}).items()}}
        return normalize(self.robot,health,sensor)
    def close(self):
        if self.client:self.client.close()


def evaluate(robot,step,samples,confirmed,*,trial=None):
    checks=[]
    def check(label,ok,action):checks.append({'label':label,'pass':bool(ok),'action':'' if ok else action})
    valid=[s for s in samples if not s.get('read_error')]
    check('Подтверждение оператора',confirmed,'Выполните шаг и подтвердите наблюдаемый результат')
    if step=='recording':
        check('Новая пробная запись прочитана обратно',trial is not None and trial.get('pass',False),'Сохраните новый показ без замечаний проверки качества')
    else:
        check('Есть свежие измерения',len(valid)>=3 and valid[-1]['at']-valid[0]['at']>=.5,'Включите управление и повторите сбор показаний')
        if valid:
            check('Нет ошибки связи или контроллера',not valid[-1].get('fault'),'Устраните ошибку связи и повторите шаг; физический STOP не выводится из закрытия сеанса')
            commands=[s.get('command') for s in valid if vector(s.get('command'))]
            moving=any(any(abs(v)>.01 for v in c) for c in commands)
            stopped=bool(commands and all(abs(v)<.01 for v in commands[-1]))
            if step=='charger':
                volts=[s['battery_v'] for s in valid if s.get('battery_fresh') and finite(s.get('battery_v'))]
                check('Зарядник отключён по подтверждению, напряжение <8.3 В и не растёт',
                    len(volts)>=5 and valid[-1]['at']-valid[0]['at']>=8 and max(volts)<8.3 and volts[-1]-volts[0]<=.05,
                    'Отсоедините зарядник и снимайте напряжение не менее 10 секунд; одного напряжения недостаточно без вашего подтверждения')
            elif step in {'forward','backward','left','right'}:
                axis=0 if step in {'forward','backward'} else 1
                sign=1 if step in {'forward','left'} else -1
                check('Зафиксирована команда нужного направления и отпускание',
                    any(len(c)>axis and c[axis]*sign>.01 for c in commands) and stopped,
                    'Повторите короткое движение указанной кнопкой и отпустите её')
                if robot=='rosmaster':
                    check('Есть измеренная скорость колёс',any(vector(s.get('measured_velocity')) and any(abs(v)>.01 for v in s['measured_velocity']) for s in valid),
                          'Проверьте обратную связь колёс ROSMASTER')
            elif step in {'joints','leader','gripper'}:
                arms=[s['arm'] for s in valid if vector(s.get('arm')) and len(s['arm'])==6]
                indices=[5] if step=='gripper' else range(5)
                check('Измерения суставов действительно изменились',len(arms)>=2 and all(max(a[i] for a in arms)-min(a[i] for a in arms)>1 for i in indices),
                    'Проверьте небольшое движение каждого требуемого сустава и обратную связь')
            elif step in {'release','stop','hidden','base_stop'}:
                arms=[s['arm'] for s in valid if vector(s.get('arm')) and len(s['arm'])==6]
                arm_moved=len(arms)>=2 and any(max(a[i] for a in arms)-min(a[i] for a in arms)>1 for i in range(6))
                check('До остановки было движение/изменение измерений',
                    moving if step in {'release','base_stop'} or (step=='hidden' and robot!='so101') else moving or arm_moved,'Начните небольшое движение перед проверкой остановки')
                if step=='release':check('После отпускания команда нулевая',stopped,'Отпустите клавиши и проверьте нулевую команду')
                else:
                    check('Управление выключено без самовозобновления',
                        len(valid)>=3 and all(not s.get('armed',True) for s in valid[-3:]),
                        'Нажмите STOP и подождите секунду; при продолжении движения выключите питание')
            elif step=='straight':
                check('Зафиксированы движение и остановка',moving and stopped,'Проедьте размеченный метр вперёд и остановитесь')
                yaw=0.;gaps=False;count=0
                for a,b in zip(valid,valid[1:]):
                    dt=b['at']-a['at']
                    if dt>.5:gaps=True
                    if finite(a.get('yaw_rate_rad_s')) and finite(b.get('yaw_rate_rad_s')):
                        yaw+=(a['yaw_rate_rad_s']+b['yaw_rate_rad_s'])*.5*dt;count+=1
                degrees=math.degrees(yaw)
                check('IMU записана без больших пропусков',count>=5 and not gaps,'Повторите измерение при устойчивой связи с IMU')
                check('Отклонение за подтверждённый оператором 1 м не более 10°',abs(degrees)<=10,
                    'Нужна штатная калибровка IMU/прямолинейности Hiwonder; параметры автоматически не меняются')
                checks.append({'label':'Интеграл IMU, градусов (не одометрия колёс)','value':degrees,'pass':True,'action':''})
            elif step=='lidar':
                check('Лидар сработал, команда стала нулевой',any('Препятствие' in str(s.get('guard')) and s.get('scan_fresh') for s in valid) and stopped,
                    'Проверьте, что коробка находится в плоскости лидара, и повторите на малой скорости')
            elif step=='sensors':
                needed=['camera_fresh','scan_fresh','depth_fresh','imu_fresh']+(['rear_scan_fresh'] if robot=='rosmaster' else [])
                check('Все требуемые потоки свежие',all(valid[-1].get(k) for k in needed),'Проверьте подключение камеры, глубины, IMU и лидаров')
                check('Кадры обновляются',len({s.get('camera_sequence') for s in valid if s.get('camera_sequence') is not None})>=2,'Проверьте, что камера передаёт новые кадры')
            elif step=='ownership':check('SO-101 принадлежит Explorer',valid[-1].get('owner')=='explorer','Нажмите «Взять SO-101 в Explorer» и подтвердите передачу')
            elif step=='ready':check('Агент получил управление',valid[-1].get('hardware_open',False),'Включите управление в панели ROSMASTER')
    return {'result':'PASS' if all(c['pass'] for c in checks) else 'FAIL','checks':checks,
            'operator_confirmed':bool(confirmed),'scope':'operator_observation_plus_read_only_telemetry',
            'motor_acknowledgement':False,'measured_wheel_feedback':robot=='rosmaster'}


class Run:
    def __init__(self,robot,reader_factory):
        if robot not in STEPS:raise ValueError('Неизвестный робот')
        self.robot=robot;self.reader_factory=reader_factory;self.samples=[];self.current=None
        self.done=threading.Event();self.thread=None;self.lock=threading.Lock()
        self.created=time.time();self.results={};self.history=[]
        self.path=DATA/'validation/physical'/f"{robot}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}.json"
    def start(self,step):
        if step not in dict(STEPS[self.robot]):raise ValueError('Неизвестный шаг')
        if self.thread and self.thread.is_alive():raise RuntimeError('Сначала закончите текущий шаг')
        if self.robot=='tank' and step not in {'charger','recording'} and self.results.get('charger',{}).get('result')!='PASS':
            raise ValueError('Сначала подтвердите отключённый зарядник и проверьте напряжение')
        self.current=step;self.samples=[];self.done.clear()
        self.thread=threading.Thread(target=self.capture,daemon=True);self.thread.start()
    def capture(self):
        reader=None;deadline=time.monotonic()+120
        try:
            reader=self.reader_factory()
            while not self.done.is_set() and time.monotonic()<deadline:
                try:sample=reader.read()
                except Exception as exc:sample={'at':time.time(),'read_error':str(exc)}
                with self.lock:self.samples.append(sample)
                self.done.wait(.1)
        except Exception as exc:
            with self.lock:self.samples.append({'at':time.time(),'read_error':str(exc)})
        finally:
            if reader:reader.close()
    def finish(self,confirmed):
        self.done.set()
        if self.thread:self.thread.join(3)
        if self.thread and self.thread.is_alive():raise RuntimeError('Дождитесь завершения чтения датчиков')
        if not self.current:raise ValueError('Сначала начните шаг')
        trial=None
        if self.current=='recording':
            from . import collection
            collection.initialize()
            from .catalog import connect
            with connect() as db:
                rows=db.execute('SELECT a.quality,a.created,s.profile FROM collection_attempts a JOIN collection_series s ON a.series=s.id WHERE a.created>?',(self.created,)).fetchall()
            trial={'pass':any(r['profile'].startswith(self.robot+'_') and json.loads(r['quality'] or '{}').get('status')=='pass' for r in rows)}
        result=evaluate(self.robot,self.current,self.samples,confirmed,trial=trial)
        self.results[self.current]={**result,'samples':list(self.samples),'finished_at':time.time()}
        self.history.append({'step':self.current,**self.results[self.current]})
        atomic_write(self.path,{'version':1,'robot':self.robot,'started_at':self.created,'steps':self.results,'attempts':self.history,
            'complete':all(self.results.get(k,{}).get('result')=='PASS' for k,_ in STEPS[self.robot]),
            'physical_commands_from_wizard':0,'recordings_or_training_started_automatically':False})
        self.current=None
        return result
