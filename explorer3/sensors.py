"""Read-only sensor packets, lossless archives and explicit policy features.

Packets are asynchronous observations, not synchronized exposures. Source stamps,
arrival ages and missing-data masks survive both recording and feature extraction.
No command endpoint is used here.
"""
import io
import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import httpx
import numpy as np

from .paths import DATA

SCHEMA='tank_environment_v1'
CHANNELS=([f'lidar_min_m_{i:02}' for i in range(36)]+
          [f'lidar_valid_fraction_{i:02}' for i in range(36)]+
          [f'depth_m_{y}_{x}' for y in range(6) for x in range(8)]+
          [f'depth_valid_fraction_{y}_{x}' for y in range(6) for x in range(8)]+
          ['gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z','imu_valid']+
          ['lidar_age_s','depth_age_s','imu_age_s'])
ROSMASTER_CHANNELS=CHANNELS+([f'rear_lidar_min_m_{i:02}' for i in range(36)]+
    [f'rear_lidar_valid_fraction_{i:02}' for i in range(36)]+['rear_lidar_age_s']+
    ['mag_x_t','mag_y_t','mag_z_t','mag_valid']+
    ['odom_x_m','odom_y_m','odom_qx','odom_qy','odom_qz','odom_qw','odom_vx','odom_vy','odom_wz','odom_valid'])


def feature_schema(robot):
    if robot not in ('tank','rosmaster'):raise ValueError('Unknown sensor robot')
    return (SCHEMA,CHANNELS) if robot=='tank' else ('rosmaster_environment_v1',ROSMASTER_CHANNELS)


def decode_packet(payload, received_ns=None, elapsed_s=0.):
    if len(payload)>16*1024*1024:raise ValueError('Sensor packet too large')
    import zipfile
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(v.file_size for v in archive.infolist())>32*1024*1024:
            raise ValueError('Expanded sensor packet too large')
    with np.load(io.BytesIO(payload),allow_pickle=False) as packet:
        meta=json.loads(packet['metadata'].tobytes())
        if meta.get('version')!=1 or meta.get('robot') not in ('tank','rosmaster'):raise ValueError('Unknown sensor schema')
        arrays={k:packet[k].copy() for k in packet.files if k!='metadata'}
    if 'rgb_jpeg' in arrays:
        import cv2
        decoded=cv2.imdecode(arrays.pop('rgb_jpeg'),cv2.IMREAD_COLOR)
        if decoded is None:raise ValueError('Invalid RGB JPEG')
        arrays['rgb']=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
    return {'meta':meta,'arrays':arrays,'payload':payload,
            'received_ns':received_ns if received_ns is not None else time.monotonic_ns(),
            'request_elapsed_s':elapsed_s}


def stream_age(packet,name,now_ns=None):
    stream=packet['meta']['streams'].get(name)
    if not stream:return float('inf')
    now=time.monotonic_ns() if now_ns is None else now_ns
    # Whole request duration is a conservative upper bound, not a clock sync.
    return stream['age_s']+packet['request_elapsed_s']+max(0,(now-packet['received_ns'])/1e9)


def environment_features(packet,now_ns=None):
    streams=packet['meta']['streams'];arrays=packet['arrays']
    ranges=np.zeros(36,np.float32);coverage=np.zeros(36,np.float32)
    depth=np.zeros((6,8),np.float32);valid_depth=np.zeros_like(depth)
    imu=np.zeros(6,np.float32);imu_valid=0.
    ages=[stream_age(packet,k,now_ns) for k in ('scan','depth','imu')]
    if ages[0]<=1. and 'scan' in arrays:
        info=streams['scan'];scan=arrays['scan']
        angles=info['angle_min']+np.arange(len(scan))*info['angle_increment']
        bins=np.floor(np.mod(angles,2*np.pi)*36/(2*np.pi)).astype(int)%36
        valid=np.isfinite(scan)&(scan>=info['range_min'])&(scan<=info['range_max'])
        for i in range(36):
            mask=bins==i;values=scan[mask&valid]
            if len(values):ranges[i]=values.min()
            coverage[i]=len(values)/max(1,mask.sum())
    if ages[1]<=1. and 'depth' in arrays:
        info=streams['depth'];raw=arrays['depth']
        if info.get('unit') not in ('mm','m'):raise ValueError('Unknown depth units')
        metric=raw.astype(np.float32)*(0.001 if info['unit']=='mm' else 1.)
        if metric.ndim!=2:raise ValueError('Depth must be a 2D metric image')
        # Driver's configured depth window, not RGB brightness or fabricated range.
        good=np.isfinite(metric)&(metric>=.15)&(metric<=4.)
        for y,rows in enumerate(np.array_split(np.arange(raw.shape[0]),6)):
            for x,cols in enumerate(np.array_split(np.arange(raw.shape[1]),8)):
                tile=metric[np.ix_(rows,cols)];mask=good[np.ix_(rows,cols)]
                if mask.any():depth[y,x]=np.median(tile[mask])
                valid_depth[y,x]=mask.mean() if mask.size else 0.
    if ages[2]<=.5 and 'imu' in arrays:
        candidate=arrays['imu']
        if candidate.shape==(6,) and np.isfinite(candidate).all():imu=candidate;imu_valid=1.
    result=np.concatenate((ranges,coverage,depth.ravel(),valid_depth.ravel(),imu,
                           [imu_valid],np.minimum(ages,10.))).astype(np.float32)
    if result.shape!=(len(CHANNELS),) or not np.isfinite(result).all():raise ValueError('Invalid sensor feature vector')
    if packet['meta']['robot']=='rosmaster':
        rear_packet={**packet,'meta':{**packet['meta'],'robot':'tank','streams':{**streams}},'arrays':{**arrays}}
        rear_packet['meta']['streams'].pop('scan',None);rear_packet['arrays'].pop('scan',None)
        if 'scan_rear' in streams:
            rear_packet['meta']['streams']['scan']=streams['scan_rear']
            if 'scan_rear' in arrays:rear_packet['arrays']['scan']=arrays['scan_rear']
        rear=environment_features(rear_packet,now_ns)
        extra=[]
        for key,size in [('mag',3),('odometry',9)]:
            value=arrays.get(key)
            good=value is not None and value.shape==(size,) and np.isfinite(value).all() and stream_age(packet,key,now_ns)<1.
            extra.extend(value.tolist() if good else [0.]*size);extra.append(float(good))
        result=np.concatenate((result,rear[:72],rear[-3:-2],extra)).astype(np.float32)
    return result


class SensorClient:
    def __init__(self,isolated=False,robot='tank'):
        from .tank import config
        from .local_work import hardware_allowed,homebrain_only
        permit=DATA/'configs/sensor-observation-enabled.json'
        extra=json.loads(permit.read_text()).get('robots',[]) if permit.exists() else []
        if homebrain_only() or not (hardware_allowed(robot) or robot in extra):raise PermissionError('Наблюдение этого робота ещё не разрешено')
        if robot not in ('tank','rosmaster'):raise ValueError('Unknown robot')
        self.robot=robot
        cfg=config() if robot=='tank' else json.loads((DATA/'configs/rosmaster-agent.json').read_text())
        port=(8043 if robot=='tank' else 8053)+int(bool(isolated))
        self.http=httpx.Client(base_url=f"http://{cfg['host']}:{port}",headers={'X-Explorer-Token':cfg['token']},
                               timeout=3.,trust_env=False)

    def snapshot(self):
        start=time.monotonic()
        response=self.http.get('/snapshot.npz');response.raise_for_status()
        received=time.monotonic_ns()
        packet=decode_packet(response.content,received,time.monotonic()-start)
        if packet['meta']['robot']!=self.robot:raise ValueError('Sensor packet came from another robot')
        return packet

    def close(self):self.http.close()

    def clock_sample(self):
        before=time.monotonic_ns();r=self.http.get('/health');after=time.monotonic_ns()
        r.raise_for_status();meta=r.json()
        if meta.get('robot')!=self.robot:raise ValueError('Clock belongs to another robot')
        return {'before_ns':before,'after_ns':after,'remote':meta}


class SensorFeed:
    """One bounded-rate reader; network and compression stay outside control loops."""
    def __init__(self,interval=.5,isolated=False,robot='tank',retain_history=False):
        self.interval=interval;self.latest=None;self.error=None;self.thread=None
        self.isolated=isolated
        self.robot=robot
        self.history=deque(maxlen=12 if retain_history else 1)
        self.history_lock=threading.Lock()
        self.done=threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.latest=None;self.error=None
        with self.history_lock:self.history.clear()
        self.done.clear();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def _run(self):
        client=None;alternate=None
        try:
            client=SensorClient(isolated=self.isolated,robot=self.robot)
            if self.isolated is None:alternate=SensorClient(isolated=True,robot=self.robot)
            while not self.done.is_set():
                start=time.monotonic()
                try:
                    try:
                        packet=client.snapshot()
                        if alternate and stream_age(packet,'rgb')>2:packet=alternate.snapshot()
                    except Exception:
                        if alternate:packet=alternate.snapshot()
                        else:raise
                    with self.history_lock:self.history.append(packet)
                    self.latest=packet;self.error=None
                except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
                self.done.wait(max(.05,self.interval-(time.monotonic()-start)))
        except Exception as exc:self.error=str(exc)
        finally:
            if client:client.close()
            if alternate:alternate.close()

    def close(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=4)

    def at_or_before(self,observation_ns):
        with self.history_lock:
            return next((p for p in reversed(self.history) if p['received_ns']<=observation_ns),None)


class SensorArchive:
    """Lossless diagnostic record, explicitly excluded from action demonstrations."""
    def __init__(self,root=None,robot='tank'):
        self.lock=threading.RLock()
        schema,channels=feature_schema(robot)
        self.root=Path(root) if root else DATA/'sensor-recordings'/(robot+'-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True,exist_ok=False)
        self.count=0;self.last_key=None;self.closed=False
        self.manifest={'version':1,'robot':robot,'category':'sensor_diagnostic','training_eligible':False,
                       'reason':'No action labels or physical task outcome','feature_schema':schema,
                       'feature_channels':channels,'created_at':time.time(),'frames':0,
                       'clock_alignment':'asynchronous latest samples; source stamps and conservative ages preserved'}
        self._write_manifest()

    def _write_manifest(self):
        temp=self.root/'manifest.tmp';temp.write_text(json.dumps(self.manifest,ensure_ascii=False,indent=2));temp.replace(self.root/'manifest.json')

    def add(self,packet):
        with self.lock:return self._add(packet)

    def _add(self,packet):
        if self.closed:raise RuntimeError('Archive closed')
        if packet['meta']['robot']!=self.manifest['robot']:raise ValueError('Robot/archive mismatch')
        key=(packet['meta']['boot_id'],packet['meta']['monotonic_ns'])
        if key==self.last_key:return False
        features=environment_features(packet,packet['received_ns'])
        ident=f'{self.count:06}'
        (self.root/(ident+'.npz')).write_bytes(packet['payload'])
        record={'file':ident+'.npz','homebrain_received_ns':packet['received_ns'],
                'request_elapsed_s':packet['request_elapsed_s'],'feature_schema':self.manifest['feature_schema'],
                'observation.environment_state':features.tolist()}
        with (self.root/'index.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        self.count+=1;self.last_key=key;self.manifest['frames']=self.count;self._write_manifest();return True

    def finish(self):
        with self.lock:
            self.closed=True;self.manifest['closed_at']=time.time();self._write_manifest();return str(self.root)


def archive_entries(robot='tank'):
    root=DATA/'sensor-recordings'
    return [json.loads(p.read_text())|{'id':p.parent.name} for p in
            sorted(root.glob(robot+'-*/manifest.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]]


def open_archive(ident):
    import re
    if not re.fullmatch(r'(tank|rosmaster)-[a-f0-9]{32}',ident):
        raise ValueError('Invalid sensor archive ID')
    root=DATA/'sensor-recordings'/ident
    if not root.resolve().is_relative_to((DATA/'sensor-recordings').resolve()):raise ValueError('Invalid archive path')
    rows=[json.loads(line) for line in (root/'index.jsonl').read_text().splitlines()]
    return root,rows


def archive_packet(root,row):
    path=(root/row['file']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Invalid sensor packet path')
    return decode_packet(path.read_bytes(),row['homebrain_received_ns'],row['request_elapsed_s'])
