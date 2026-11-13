"""Measured clock translation and readiness of the physical observation bus.

This is not hardware exposure synchronization or a motor controller. No fabricated
world transform: each robot keeps its own frame tree until surveyed calibration.
"""
import json
import math
import re
import time
from .paths import DATA


def frame_id(robot, frame):
    if robot not in ('tank', 'rosmaster'):raise ValueError('Unknown robot')
    frame=frame.lstrip('/')
    if not frame or not re.fullmatch(r'[A-Za-z0-9_/]+',frame):raise ValueError('Invalid source frame')
    return robot+'/'+frame


class ClockAlignment:
    """Midpoint estimate with RTT bound; refreshed, boot-scoped and jump-aware.

    The drift allowance is a conservative engineering budget (250 ppm), not an
    oscillator calibration. Precision is reported, never assumed from NTP status.
    """
    def __init__(self):
        self.boot=None;self.samples=[];self.ros_offset=None;self.wall_offset=None
        self.generation=0

    def add(self,sample,wall_offset_ns=None):
        before=sample['before_ns'];after=sample['after_ns'];meta=sample['remote']
        pair=meta.get('clock_pair') or {}
        if after<before or after-before>2_000_000_000:raise ValueError('Invalid clock round trip')
        if pair.get('use_sim_time') or not pair.get('ros_time_ns'):raise ValueError('Physical ROS clock unavailable')
        remote=pair['monotonic_ns'];ros_offset=pair['ros_time_ns']-remote
        wall=time.time_ns()-time.monotonic_ns() if wall_offset_ns is None else wall_offset_ns
        jumped=(self.boot!=meta['boot_id'] or
                self.ros_offset is not None and abs(ros_offset-self.ros_offset)>50_000_000 or
                self.wall_offset is not None and abs(wall-self.wall_offset)>50_000_000)
        if jumped:self.samples=[];self.generation+=1
        self.boot=meta['boot_id'];self.ros_offset=ros_offset;self.wall_offset=wall
        self.samples=[s for s in self.samples if 0<=after-s['at_ns']<=15_000_000_000]
        self.samples.append({'offset_ns':(before+after)//2-remote,
                             'uncertainty_ns':(after-before)//2+pair.get('uncertainty_ns',0),
                             'at_ns':after})

    def estimate(self,meta,now_ns=None):
        now=time.monotonic_ns() if now_ns is None else now_ns
        if meta['boot_id']!=self.boot or not self.samples:raise ValueError('Clock not sampled for current boot')
        recent=[s for s in self.samples if 0<=now-s['at_ns']<=15_000_000_000]
        if not recent:raise ValueError('Clock samples expired')
        best=min(recent,key=lambda s:s['uncertainty_ns']+(now-s['at_ns'])*.00025)
        pair=meta['clock_pair']
        if pair.get('use_sim_time') or not pair.get('ros_time_ns'):raise ValueError('Invalid physical source clock')
        if abs(pair['ros_time_ns']-pair['monotonic_ns']-self.ros_offset)>50_000_000:
            raise ValueError('Source clock jumped; resampling required')
        return {**best,'uncertainty_ns':int(best['uncertainty_ns']+(now-best['at_ns'])*.00025),
                'generation':self.generation,'kind':'midpoint_estimate_not_hardware_sync'}

    def stamp(self,meta,stream,now_ns=None):
        estimate=self.estimate(meta,now_ns);pair=meta['clock_pair']
        source=stream.get('source_timestamp_ns',0)
        if source:
            delta=source-pair['ros_time_ns']
            if not -60_000_000_000<=delta<=100_000_000:raise ValueError('Source timestamp outside physical clock window')
            remote=pair['monotonic_ns']+delta;kind='source_header'
        else:
            remote=stream['received_ns'];kind='observer_arrival_no_source_header'
        return remote+estimate['offset_ns']+self.wall_offset,kind


def read_hub_status():
    path=DATA/'ros2-hub/status.json'
    try:status=json.loads(path.read_text())
    except (OSError,ValueError):return {'running':False,'robots':{},'reason':'Общая шина ещё не запущена'}
    age=max(0,time.time()-status.get('updated_at',0))
    status['running']=age<5;status['status_age_s']=age
    # Never display a frozen, old status file as fresh sensor measurements.
    for robot in status.get('robots',{}).values():
        robot['ages_s']={key:value+age for key,value in robot.get('ages_s',{}).items()}
        if 'uncertainty_ms' in robot:robot['uncertainty_ms']+=age*.25
    if not status['running']:
        status['reason']='Нет свежего состояния общей шины'
        status['pairing']={'matched':False,'reason':'Нет свежей пары кадров'}
    return status


def readiness(status,required=('tank','rosmaster'),max_uncertainty_ms=50):
    """Only observation readiness. Never grants permission to move a robot."""
    problems=[]
    if not status.get('running'):problems.append('Общая шина не отвечает')
    for robot in required:
        state=status.get('robots',{}).get(robot,{})
        if state.get('error'):problems.append(robot+': '+state['error'])
        if state.get('conversion_errors'):problems.append(robot+': часть потоков не передана в ROS 2')
        if state.get('uncertainty_ms',math.inf)>max_uncertainty_ms:problems.append(robot+': время не согласовано с нужной точностью')
        for stream in ('rgb','depth','scan','imu'):
            if state.get('ages_s',{}).get(stream,math.inf)>1.:
                problems.append(robot+': нет свежего '+stream)
    return {'observations_ready':not problems,'problems':problems,'motion_authorized':False,
            'shared_map_ready':False,'reason':'Положение роботов в общей карте ещё не откалибровано'}


def match_observations(candidates,max_skew_ms=100):
    """Find a cross-robot pair whose timestamp intervals fit the chosen budget.

    Candidates contain measured header times translated onto Homebrain time.
    Network-clock uncertainty expands the interval; no nearest-frame match is
    allowed to quietly erase that uncertainty. Returned IDs address raw packets.
    """
    if set(candidates)!= {'tank','rosmaster'}:return {'matched':False,'reason':'Ожидаются обе машины'}
    pairs=[]
    for a in candidates['tank']:
        for b in candidates['rosmaster']:
            if max(a['age_s'],b['age_s'])>1 or min(a['age_s'],b['age_s'])<-.1:continue
            skew=(abs(a['timestamp_ns']-b['timestamp_ns'])+a['uncertainty_ns']+b['uncertainty_ns'])/1e6
            pairs.append((skew,a,b))
    if not pairs:return {'matched':False,'reason':'Нет свежей пары кадров'}
    skew,a,b=min(pairs,key=lambda x:(x[0],-min(x[1]['timestamp_ns'],x[2]['timestamp_ns'])))
    return {'matched':skew<=max_skew_ms,'worst_case_skew_ms':round(skew,3),'limit_ms':max_skew_ms,
            'reason':'Сопоставлены по времени с учётом погрешности' if skew<=max_skew_ms else 'Разница времени слишком велика',
            'packets':{'tank':a['packet_id'],'rosmaster':b['packet_id']},
            'hardware_synchronized':False}
