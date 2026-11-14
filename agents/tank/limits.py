"""Robot-local bounds and sensor guards. No I/O, ROS or hardware construction."""
import json
import math
from pathlib import Path
import time

DEFAULTS={
    'version':1,'manual':{'v':.2,'w':.5},'policy':{'v':.3,'w':1.},
    'absolute':{'v':.6,'w':2.},'acceleration':{'v':.5,'w':2.},
    'command_ttl_s':.35,'lidar_stop_m':.15,'dock_stop_m':.05,
    'dock_max_speed_m_s':.1,'sector_half_angle_deg':30.,'tilt_stop_deg':20.,
    'battery_stop_v':6.9,'battery_warning_v':7.2,'sensor_max_age_s':.5,
}


def load(root):
    path=Path(root)/'configs/tank-limits.json'
    value=json.loads(path.read_text()) if path.exists() else json.loads(json.dumps(DEFAULTS))
    if value.get('version')!=1:raise ValueError('Неизвестный формат пределов Tank')
    for key,default in DEFAULTS.items():
        if isinstance(default,dict):
            for channel in default:
                number=value[key][channel]
                if isinstance(number,bool) or not math.isfinite(number) or number<=0:
                    raise ValueError('Неверный предел Tank: '+key+'/'+channel)
        elif key!='version':
            number=value[key]
            if isinstance(number,bool) or not math.isfinite(number) or number<=0:
                raise ValueError('Неверный предел Tank: '+key)
    # Config may tighten factory bounds, never override them.
    value['absolute']['v']=min(value['absolute']['v'],.6)
    value['absolute']['w']=min(value['absolute']['w'],2.)
    value['command_ttl_s']=min(value['command_ttl_s'],.35)
    return value


def clip(value,v,w,mode='manual'):
    if not all(math.isfinite(x) for x in (v,w)):raise ValueError('Нечисловая скорость Tank')
    group=value['manual'] if mode=='manual' else value['policy']
    ceilings=[min(group[k],value['absolute'][k],DEFAULTS['manual' if mode=='manual' else 'policy'][k]) for k in ('v','w')]
    return tuple(max(-cap,min(cap,x)) for x,cap in zip((v,w),ceilings))


class Ramp:
    def __init__(self,value,clock=time.monotonic):
        self.value=value;self.clock=clock;self.reset()

    def reset(self):
        self.previous=(0.,0.);self.updated=self.clock()

    def apply(self,v,w):
        now=self.clock();dt=max(0.,min(.1,now-self.updated));self.updated=now
        self.previous=tuple(0. if target==0. else old+max(-a*dt,min(a*dt,target-old))
            for target,old,a in zip((v,w),self.previous,
                (self.value['acceleration']['v'],self.value['acceleration']['w'])))
        return self.previous


def sector_distance(ranges,angle_min,increment,range_min,range_max,*,rear=False,half_angle=30.):
    selected=[];center=math.pi if rear else 0.
    for index,distance in enumerate(ranges):
        angle=angle_min+increment*index
        delta=math.atan2(math.sin(angle-center),math.cos(angle-center))
        if abs(delta)<=math.radians(half_angle) and math.isfinite(distance) and range_min<=distance<=range_max:
            selected.append(float(distance))
    return min(selected) if selected else None


def tilt_degrees(acceleration):
    if acceleration is None or len(acceleration)!=3 or not all(math.isfinite(v) for v in acceleration):return None
    x,y,z=acceleration
    if math.sqrt(x*x+y*y+z*z)<1.:return None
    return math.degrees(math.atan2(math.hypot(x,y),z))


def guard(value,v,w,*,battery,battery_age,tilt,imu_age,front,rear,scan_age,profile='tank_dock'):
    """Return an immediate zero when blocked; reverse escape remains available."""
    if v==0. and w==0.:return (0.,0.),None
    reason=None
    if battery is None or not math.isfinite(battery) or battery_age>2.:reason='Нет свежего напряжения батареи'
    elif battery<value['battery_stop_v']:reason='Батарея ниже 6,9 В: зарядите выключенный Tank'
    elif tilt is None or imu_age>value['sensor_max_age_s']:reason='Нет свежего наклона IMU'
    elif tilt>value['tilt_stop_deg']:reason='Наклон больше 20°: поставьте Tank ровно'
    elif scan_age>value['sensor_max_age_s']:reason='Нет свежего скана лидара'
    else:
        distance=front if v>0 else rear if v<0 else None
        threshold=value['dock_stop_m'] if profile.startswith('tank_dock') and abs(v)<=value['dock_max_speed_m_s'] and abs(w)<=.5 else value['lidar_stop_m']
        if v!=0 and distance is None:reason='Лидар не видит направление движения'
        elif distance is not None and distance<threshold:reason='Препятствие впереди' if v>0 else 'Препятствие сзади'
    return ((0.,0.) if reason else (v,w)),reason
