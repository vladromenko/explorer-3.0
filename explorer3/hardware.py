"""Explicit-start SO-101 I/O. Importing this module never opens a device.

No foreign project code is imported. Bus commands use the private LeRobot environment.
A bus write is attempted once; a failure terminates the session, never retries motion.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import termios
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .resources import ResourceLease, assert_hardware_available
from .paths import DATA

JOINTS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
# A relaxed joint can read a few ticks beyond an already calibrated end stop.
# This is a feedback deadband, not an extension of commanded joint limits.
ENCODER_BOUNDARY_TOLERANCE_TICKS = 8
ROOT = DATA
DEFAULT_CAMERAS = {
    'wrist': '/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera-video-index0',
}
_cv_setup_lock=threading.Lock()
_cv_ready=False

def configure_cv():
    """Avoid tiny-image CPU oversubscription; configure before starting CV threads."""
    global _cv_ready
    with _cv_setup_lock:
        if not _cv_ready:
            import cv2
            cv2.setNumThreads(1)
            _cv_ready=True


def hardware_settings(settings: dict | None = None) -> dict:
    settings = dict(settings or {})
    result = {
        'leader_port': '/dev/so101-leader', 'follower_port': '/dev/so101-follower',
        'cameras': dict(DEFAULT_CAMERAS), 'width': 640, 'height': 480,
        'camera_formats': {'top':'MJPG','wrist':'YUYV'},
        'camera_mode': 'wrist_only',
        'fps': 30, 'max_camera_age_ms': 100, 'max_step_degrees': 2.0,
        'max_gripper_step': 3.0, 'max_speed_degrees_s': 45.0,
        'smooth_recording_actions': False,
        'teleoperation_control_hz': 60,
        'calibration_dir': str(ROOT / 'calibration'),
    }
    for key in result:
        if key in settings: result[key] = settings[key]
    if result['camera_mode'] not in ('two_camera','wrist_only'):
        raise ValueError('Режим камер должен быть two_camera или wrist_only')
    expected_cameras = ('top','wrist') if result['camera_mode']=='two_camera' else ('wrist',)
    result['cameras']={name:result['cameras'][name] for name in expected_cameras if name in result['cameras']}
    result['camera_formats']={name:result['camera_formats'][name] for name in expected_cameras if name in result['camera_formats']}
    for key in ('fps','max_camera_age_ms','max_step_degrees','max_gripper_step','max_speed_degrees_s','teleoperation_control_hz'):
        result[key] = float(result[key])
        if not math.isfinite(result[key]): raise ValueError(f'{key}: требуется конечное число')
    for key in ('width','height'): result[key] = int(result[key])
    if not isinstance(result['smooth_recording_actions'],bool):raise ValueError('smooth_recording_actions должен быть bool')
    if not 1 <= float(result['fps']) <= 30: raise ValueError('FPS должен быть в пределах 1–30')
    if not 10<=result['teleoperation_control_hz']<=120:raise ValueError('Частота телеоперации: 10–120 Hz')
    if (int(result['width']),int(result['height'])) != (640,480):
        raise ValueError('Проверяемый режим записи: 640×480; другая геометрия требует отдельной проверки')
    if not 0 < float(result['max_step_degrees']) <= 2.0: raise ValueError('Шаг сустава должен быть 0–2°')
    if not 0 < float(result['max_gripper_step']) <= 5: raise ValueError('Шаг gripper должен быть 0–5%')
    if not 0 < float(result['max_speed_degrees_s']) <= 60: raise ValueError('Скорость должна быть 0–60°/s')
    if not 0 < result['max_camera_age_ms'] <= 500: raise ValueError('Допустимый возраст камеры: 0–500 ms')
    if tuple(result['cameras']) != expected_cameras: raise ValueError('Для выбранного режима отсутствует камера: '+', '.join(expected_cameras))
    if tuple(result['camera_formats']) != expected_cameras or any(value not in ('MJPG','YUYV') for value in result['camera_formats'].values()):
        raise ValueError('Формат выбранных камер должен быть MJPG или YUYV')
    paths = [result['leader_port'],result['follower_port'],*result['cameras'].values()]
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError('Leader, follower и камеры должны использовать разные устройства')
    return result


def hardware_paths(settings: dict | None = None, leader: bool = True, include_cameras: bool = True) -> list[str]:
    cfg = hardware_settings(settings)
    paths = [cfg['follower_port']]
    if include_cameras:paths.extend(cfg['cameras'].values())
    if leader: paths.append(cfg['leader_port'])
    return paths


def calibration_snapshot(calibration_dir: str | Path | None = None) -> dict:
    directory = Path(calibration_dir or ROOT / 'calibration').resolve()
    # Calibration must be an independent copy inside the Explorer data directory.
    if not directory.is_relative_to(ROOT): raise ValueError('Калибровка должна быть в собственном каталоге Explorer')
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = dict(manifest)
    result['values'] = {}
    for role in ('leader','follower'):
        entry = manifest['files'][role]
        path = (directory / entry['path']).resolve()
        if not path.is_relative_to(directory): raise ValueError('Некорректный путь calibration snapshot')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'Изменён calibration snapshot {role}: hash не совпал')
        values = json.loads(raw)
        if set(values) != set(JOINTS): raise ValueError('Неверная схема motor calibration')
        for i, name in enumerate(JOINTS, 1):
            if values[name]['id'] != i or values[name]['range_min'] >= values[name]['range_max']:
                raise ValueError('Неверные ID/ranges motor calibration')
        result['values'][role] = values
    return result


class CameraCapture:
    """Own server camera reader; never adjusts exposure/focus/white balance controls."""
    def __init__(self, name: str, device: str, cfg: dict):
        self.name, self.device, self.cfg = name, device, cfg
        self.frames = deque(maxlen=8)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.sequence = 0
        self.cap = None
        self.thread = None
        self.info = {}

    def start(self):
        if (DATA/'configs/so101-external-owner.json').exists():
            raise PermissionError('Камера оставлена прежнему управлению')
        configure_cv()
        import cv2
        assert_hardware_available([self.device])
        self.cap = cv2.VideoCapture(str(self.device), cv2.CAP_V4L2)
        try:
            if not self.cap.isOpened(): raise RuntimeError(f'Камера {self.name} не открылась: {self.device}')
            # Stream mode is selected only after explicit user start; no autofocus/controls writes.
            pixel_format=self.cfg.get('camera_formats',{'top':'MJPG','wrist':'YUYV'})[self.name]
            mode_properties = [('fourcc',cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*pixel_format)),
                               ('width',cv2.CAP_PROP_FRAME_WIDTH,int(self.cfg['width'])),
                               ('height',cv2.CAP_PROP_FRAME_HEIGHT,int(self.cfg['height'])),
                               ('fps',cv2.CAP_PROP_FPS,float(self.cfg['fps']))]
            control_properties = {name:getattr(cv2,'CAP_PROP_'+name) for name in
                                  ('AUTOFOCUS','FOCUS','AUTO_EXPOSURE','EXPOSURE','AUTO_WB','WB_TEMPERATURE')
                                  if hasattr(cv2,'CAP_PROP_'+name)}
            controls_before={name:self.cap.get(prop) for name,prop in control_properties.items()}
            stream_writes=[]
            for name,property_id,value in mode_properties:
                current=self.cap.get(property_id)
                if not math.isfinite(current) or abs(current-value)>(max(.5,abs(value)*.01) if name=='fps' else .1):
                    self.cap.set(property_id,value)
                    stream_writes.append(name)
            controls_after={name:self.cap.get(prop) for name,prop in control_properties.items()}
            actual_fourcc=int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.info = {
                'device': self.device, 'device_resolved': str(Path(self.device).resolve()),
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps_reported': self.cap.get(cv2.CAP_PROP_FPS),
                'pixel_format_requested':pixel_format,
                'pixel_format_reported':''.join(chr((actual_fourcc >> (8*i)) & 255) for i in range(4)),
                'timestamp_basis': 'host_monotonic_capture_read_end; exposure timestamp unavailable',
                'controls_written': [], 'camera_geometry_valid': False,
                'stream_configuration_writes':stream_writes,'controls_readback_before':controls_before,
                'controls_readback_after':controls_after,
                'controls_changed_by_driver':[name for name in controls_before if controls_before[name]!=controls_after[name]],
                'controls_restored':False,
            }
            if (self.info['width'],self.info['height']) != (self.cfg['width'],self.cfg['height']):
                raise RuntimeError(f'Камера {self.name}: фактическое разрешение {self.info["width"]}x{self.info["height"]} не совпало')
            if self.info['pixel_format_reported']!=pixel_format:
                raise RuntimeError(f'Камера {self.name}: запрошен {pixel_format}, получен {self.info["pixel_format_reported"]}')
            actual_fps=self.info['fps_reported']
            if not math.isfinite(actual_fps) or abs(actual_fps-self.cfg['fps'])>max(.5,self.cfg['fps']*.05):
                raise RuntimeError(f'Камера {self.name}: FPS readback {actual_fps:g} вместо {self.cfg["fps"]:g}; поток не запущен')
            self.thread = threading.Thread(target=self._run, name='so101-ai-camera-'+self.name, daemon=True)
            self.thread.start()
            deadline = time.monotonic()+3
            while self.sequence<2 and not self.error and time.monotonic()<deadline:
                self.stop_event.wait(.01)
            if self.sequence<2: raise RuntimeError(f'Нет стабильного потока {self.name}: {self.error or "менее двух кадров за 3s"}')
        except Exception:
            self.close()
            raise

    def _run(self):
        import cv2
        try:
            while not self.stop_event.is_set():
                read_start = time.monotonic_ns()
                ok,bgr = self.cap.read()
                read_end = time.monotonic_ns()
                if not ok or bgr is None: raise RuntimeError('V4L2 read returned no image')
                rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                if rgb.shape != (self.cfg['height'],self.cfg['width'],3):
                    raise RuntimeError('Размер камеры изменился во время записи')
                rgb.flags.writeable = False
                self.sequence += 1
                frame = {'image':rgb,'timestamp_ns':read_end,'read_start_ns':read_start,
                         'read_end_ns':read_end,'sequence':self.sequence}
                with self.lock: self.frames.append(frame)
                self.info['capture_frames']=self.sequence
                if len(self.frames)>1:
                    duration=(read_end-self.frames[0]['timestamp_ns'])/1e9
                    self.info['fps_observed_recent']=(len(self.frames)-1)/duration if duration>0 else None
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.stop_event.set()

    def latest_before(self, cutoff_ns):
        if self.error: raise RuntimeError(f'Камера {self.name}: {self.error}')
        with self.lock:
            for frame in reversed(self.frames):
                if frame['timestamp_ns'] <= cutoff_ns: return dict(frame)
        return None

    def close(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.thread and self.thread.is_alive():
            self.error = 'camera reader did not stop within 2s; session stopped'


class HardwareSource:
    """Two-camera leader→follower teleoperation, or follower-only policy I/O.

    Constructing the object is inert. `start(explicit_user_start=True)` is mandatory.
    """
    source = 'hardware'
    def __init__(self, settings: dict | None = None, *, teleoperate: bool = True, capture_cameras: bool = True, camera_provider=None):
        self.cfg = hardware_settings(settings)
        self.teleoperate = teleoperate
        self.capture_cameras = capture_cameras
        self.camera_provider=camera_provider
        self.camera_borrow_token=None
        self.calibration = calibration_snapshot(self.cfg['calibration_dir'])
        self.cameras: dict[str,CameraCapture] = {}
        self.buses: dict[str,Any] = {}
        self.lease = None
        self.camera_lease = None
        self.started = False
        self.last_command_ns = None
        self.last_camera_sequences = {}
        self.closed = False
        self.verified_calibration = {}
        self._torque_touched = set()
        self._close_errors = []
        self.startup_boundary_adjustments = {}
        self.last_teleop_target = None
        self.teleop_phase = 'aligning'
        self._stop_requested=None

    def start(self, *, explicit_user_start: bool = False, stop=None):
        if not explicit_user_start: raise PermissionError('Нужен явный запуск записи/политики в UI')
        if (DATA/'configs/so101-external-owner.json').exists():
            raise PermissionError('Follower и камера оставлены прежнему управлению; полный SO-101 профиль заблокирован')
        def check_cancelled():
            if stop and stop(): raise InterruptedError('Запуск hardware отменён пользователем до дальнейших команд')
        check_cancelled()
        self._stop_requested=stop
        if self.started: raise RuntimeError('Source already started')
        self.closed = False
        self._close_errors = []
        self._torque_touched = set()
        self.buses = {}
        self.cameras = {}
        self.verified_calibration = {}
        self.last_camera_sequences = {}
        lease = ResourceLease('hardware')
        lease.__enter__()
        self.lease = lease
        try:
            assert_hardware_available(hardware_paths(self.cfg,leader=self.teleoperate,include_cameras=self.capture_cameras))
            if self.capture_cameras and self.camera_provider is not None:
                self.camera_borrow_token,self.cameras=self.camera_provider.acquire_for_recording(self.cfg)
            elif self.capture_cameras:
                camera_lease=ResourceLease('cameras')
                camera_lease.__enter__()
                self.camera_lease=camera_lease
            # Imports resolve only to the laboratory .venv, never a foreign worktree.
            from lerobot.motors import Motor, MotorCalibration, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
            for name,device in (self.cfg['cameras'].items() if self.capture_cameras and self.camera_provider is None else []):
                check_cancelled()
                camera = CameraCapture(name,device,self.cfg)
                self.cameras[name] = camera
                camera.start()
            for role in (('leader','follower') if self.teleoperate else ('follower',)):
                check_cancelled()
                device = self.cfg[role+'_port']
                assert_hardware_available([device])
                motors = {name:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if name=='gripper' else MotorNormMode.DEGREES)
                          for i,name in enumerate(JOINTS,1)}
                calibration = {name:MotorCalibration(**v) for name,v in self.calibration['values'][role].items()}
                bus = FeetechMotorsBus(port=device,motors=motors,calibration=calibration)
                self.buses[role] = bus
                bus.connect(handshake=False)
                fcntl.ioctl(bus.port_handler.ser.fileno(),termios.TIOCEXCL)
                bus._handshake()
                actual = bus.read_calibration()
                mismatches = [n for n in JOINTS if any(getattr(actual[n],k) != getattr(calibration[n],k)
                              for k in ('id','homing_offset','range_min','range_max'))]
                if mismatches: raise RuntimeError(f'{role}: калибровка в моторах отличается от snapshot: {mismatches}; запись не запущена')
                for name in JOINTS:
                    if int(bus.read('Operating_Mode',name,normalize=False,num_retry=0)) != 0:
                        raise RuntimeError(f'{role}/{name}: нужен Position mode; автоматическая перенастройка запрещена')
                self.verified_calibration[role] = {'verified_live':True,'timestamp_ns':time.monotonic_ns(),
                                                  'snapshot_sha256':self.calibration['files'][role]['sha256']}
            # Explicit start authorizes torque. Seed the present raw position before enabling it.
            check_cancelled()
            now=time.monotonic_ns()
            for name,camera in self.cameras.items():
                latest=camera.latest_before(now)
                if latest is None or now-latest['timestamp_ns']>self.cfg['max_camera_age_ms']*1e6:
                    raise RuntimeError(f'Камера {name} stale до включения torque; Start отменён')
            follower = self.buses['follower']
            self._torque_touched.add('follower')
            follower.disable_torque(num_retry=0)
            present_raw = follower.sync_read('Present_Position',normalize=False,num_retry=1)
            hold_raw = self._startup_hold_raw(present_raw)
            present_normalized=follower._normalize({follower.motors[name].id:value for name,value in hold_raw.items()})
            present=[present_normalized[follower.motors[name].id] for name in JOINTS]
            self._safe_targets(present,present,time.monotonic_ns())
            check_cancelled()
            follower.sync_write('Goal_Position',hold_raw,normalize=False,num_retry=0)
            self.last_teleop_target=list(present)
            self.teleop_phase='aligning'
            if self.teleoperate:
                self._torque_touched.add('leader')
                self.buses['leader'].disable_torque(num_retry=0)
            check_cancelled()
            follower.enable_torque(num_retry=0)
            check_cancelled()
            self.started = True
            self.last_command_ns = time.monotonic_ns()
            return self.provenance()
        except Exception:
            self.close()
            raise

    def provenance(self):
        return {'source':self.source,'lerobot_version':importlib.metadata.version('lerobot'),
                'joints':list(JOINTS),'units':['degrees']*5+['percent_calibrated_range'],
                'calibration_snapshot':self.calibration,'live_calibration':self.verified_calibration,
                'cameras':{n:c.info for n,c in self.cameras.items()},
                'feedback_unavailable':['velocity','current','load','force'],
                'writes_acknowledgement':'Goal_Position sync_write has no per-motor ACK; actual transmitted targets, not measured achievement',
                'encoder_boundary_tolerance_ticks':ENCODER_BOUNDARY_TOLERANCE_TICKS,
                'camera_capture_enabled':self.capture_cameras,
                'teleoperation_profile':('bounded smooth calibrated LeRobot leader action' if self.cfg['smooth_recording_actions'] else 'bounded command-space initial alignment, then direct calibrated LeRobot leader action') if self.teleoperate else None,
                'startup_boundary_adjustments':self.startup_boundary_adjustments,
                'motion_retry_count':0,'camera_geometry_valid':False,'wrist_hand_eye_valid':False}

    def _startup_hold_raw(self, present_raw):
        """Keep calibrated registers intact; seed an in-range hold within ≤8 ticks.

        Reject larger excursions. The operator's explicit Start authorizes this
        small hold correction; subsequent targets still follow bounded leader
        positions with no single jump to a distant starting pose.
        """
        if set(present_raw)!=set(JOINTS):raise ValueError('Нет положения одного из моторов follower')
        targets={}
        self.startup_boundary_adjustments={}
        for name,value in present_raw.items():
            cal=self.calibration['values']['follower'][name]
            if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=4095:
                raise RuntimeError(f'follower/{name}: невалидное показание энкодера {value}; включение отменено')
            budget=self.cfg['max_gripper_step']*(cal['range_max']-cal['range_min'])/100 if name=='gripper' else self.cfg['max_step_degrees']*4095/360
            tolerance=min(ENCODER_BOUNDARY_TOLERANCE_TICKS,math.floor(budget))
            target=max(cal['range_min'],min(cal['range_max'],value))
            if abs(target-value)>tolerance:
                raise RuntimeError(f'follower/{name}: положение {value} вне сохранённого диапазона [{cal["range_min"]}, {cal["range_max"]}] на {abs(target-value)} тиков (допуск {tolerance}). Включение отменено; новая калибровка не запускалась')
            targets[name]=target
            if target!=value:self.startup_boundary_adjustments[name]={'measured_raw_ticks':value,'hold_raw_ticks':target,'difference_ticks':target-value,'calibration_changed':False}
        return targets

    def observe(self, *, require_images=True):
        if not self.started: raise RuntimeError('Hardware source is not started')
        read_start = time.monotonic_ns()
        state_dict = self.buses['follower'].sync_read('Present_Position',num_retry=1)
        read_end = time.monotonic_ns()
        images = {}
        for name,camera in (self.cameras.items() if require_images else []):
            frame = camera.latest_before(read_end)
            if frame is None: raise RuntimeError(f'Нет причинного кадра {name} до observation timestamp')
            age_ms = (read_end-frame['timestamp_ns'])/1e6
            if age_ms>self.cfg['max_camera_age_ms']:
                raise RuntimeError(f'Камера {name} stale: {age_ms:.1f} ms')
            previous = self.last_camera_sequences.get(name)
            frame['reused'] = previous == frame['sequence']
            frame['dropped_since_previous'] = max(0,frame['sequence']-(previous or frame['sequence'])-1)
            self.last_camera_sequences[name] = frame['sequence']
            images[name] = frame
        state = [float(state_dict[n]) for n in JOINTS]
        if not all(math.isfinite(v) for v in state): raise RuntimeError('Не конечная позиция суставов')
        return {'timestamp_ns':read_end,'observation_timestamp_ns':read_end,
                'state_timestamp_ns':read_end,'state_read_start_ns':read_start,'state_read_end_ns':read_end,
                'observation.state':state,'images':images,'state_valid':True,'source':'hardware',
                'measured_response_kind':'measured_joint_position'}

    def begin_episode(self):
        """Frames intentionally ignored during an inter-episode pause are not capture drops."""
        self.last_camera_sequences = {}

    def _safe_targets(self, requested, measured, now_ns):
        if len(requested)!=6 or not all(math.isfinite(float(v)) for v in requested):
            raise ValueError('Политика/leader вернула неверный action')
        if len(measured)!=6: raise ValueError('State должен содержать шесть суставов')
        dt = min(.1,max(.001,(now_ns-(self.last_command_ns or now_ns))/1e9))
        step = min(float(self.cfg['max_step_degrees']),float(self.cfg['max_speed_degrees_s'])*dt)
        result = {}
        limits = []
        for i,name in enumerate(JOINTS):
            cal = self.calibration['values']['follower'][name]
            if name=='gripper': low,high,delta = 0.,100.,float(self.cfg['max_gripper_step'])
            else:
                mid = (cal['range_min']+cal['range_max'])/2
                low,high = (cal['range_min']-mid)*360/4095,(cal['range_max']-mid)*360/4095
                # Conservative additional software bound; live collision safety remains operator responsibility.
                low,high = max(low,-175.),min(high,175.)
                delta = step
            tolerance=min(ENCODER_BOUNDARY_TOLERANCE_TICKS*360/4095,self.cfg['max_step_degrees']) if name!='gripper' else min(ENCODER_BOUNDARY_TOLERANCE_TICKS*100/(cal['range_max']-cal['range_min']),self.cfg['max_gripper_step'])
            if not math.isfinite(float(measured[i])) or not low-tolerance <= measured[i] <= high+tolerance:
                raise RuntimeError(f'{name}: measured state вне безопасных абсолютных пределов; движение остановлено')
            value = float(requested[i])
            if not self.teleoperate and not low <= value <= high:
                raise ValueError(f'{name}: policy action вне абсолютных пределов; команда отклонена')
            bounded = max(low,min(high,value))
            if not (self.teleoperate and self.teleop_phase=='following' and not self.cfg['smooth_recording_actions']):
                previous=self.last_teleop_target[i] if self.teleoperate and self.last_teleop_target is not None else measured[i]
                reference=max(low,min(high,previous))
                bounded = max(reference-delta,min(reference+delta,bounded))
                max_step=self.cfg['max_gripper_step'] if name=='gripper' else self.cfg['max_step_degrees']
                bounded=max(low,min(high,max(previous-max_step,min(previous+max_step,bounded))))
                # During initial alignment the goal advances from the actual
                # follower pose.  A slow servo must not turn an otherwise
                # valid large leader/follower offset into an immediate fault:
                # hold the commanded lead at the existing 10-unit bound and
                # let the goal advance again as feedback catches up.
                if self.teleoperate:
                    bounded=max(measured[i]-10.,min(measured[i]+10.,bounded))
            result[name] = bounded
            if abs(value-bounded)>1e-6: limits.append(name)
        return result,limits

    def send(self, requested, observation):
        if not self.started: raise RuntimeError('Hardware source is not started')
        start_ns = time.monotonic_ns()
        state_timestamp = observation['state_timestamp_ns']
        if isinstance(state_timestamp,bool) or not isinstance(state_timestamp,int) or state_timestamp <= 0:
            raise ValueError('Некорректный state_timestamp_ns')
        state_age = start_ns-state_timestamp
        if state_age < 0: raise RuntimeError('State timestamp находится в будущем; команда отклонена')
        if observation.get('state_valid',True) is not True:
            raise RuntimeError('State invalid; команда отклонена')
        if state_age > 500_000_000:
            raise RuntimeError('State слишком старый для движения (>500ms), нужна новая observation')
        safe,limited = self._safe_targets(requested,observation['observation.state'],start_ns)
        bus = self.buses['follower']
        raw_by_id = bus._unnormalize({bus.motors[n].id:v for n,v in safe.items()})
        raw = {name:raw_by_id[bus.motors[name].id] for name in JOINTS}
        # Single send, no automatic retry even after ambiguous USB/ACK failure.
        bus.sync_write('Goal_Position',raw,normalize=False,num_retry=0)
        end_ns = time.monotonic_ns()
        actual_by_id = bus._normalize(raw_by_id)
        actual = [float(actual_by_id[bus.motors[name].id]) for name in JOINTS]
        self.last_command_ns = end_ns
        if self.teleoperate:self.last_teleop_target=list(actual)
        return {'action':actual,'action_after_limits':[safe[n] for n in JOINTS],'action_raw_ticks':[raw[n] for n in JOINTS],
                'action_timestamp_ns':end_ns,'action_send_start_ns':start_ns,'action_send_end_ns':end_ns,
                'action_valid':True,'action_acknowledged':False,'action_provenance':'transmitted_quantized_goal_position',
                'limits_applied':limited}

    def sample(self, frame_id, *, stop=None, require_images=True):
        stop=stop or getattr(self,'_stop_requested',None)
        if stop and stop(): raise InterruptedError('Телеоперация остановлена до чтения')
        observation = self.observe() if require_images else self.observe(require_images=False)
        if not self.teleoperate: raise RuntimeError('sample only for leader teleoperation')
        leader_start = time.monotonic_ns()
        requested = self.buses['leader'].sync_read('Present_Position',num_retry=1)
        leader_end = time.monotonic_ns()
        if stop and stop(): raise InterruptedError('Телеоперация остановлена до отправки команды')
        requested_vector=[float(requested[n]) for n in JOINTS]
        if self.teleop_phase=='aligning' and all(abs(a-b)<=(3. if n=='gripper' else 2.) for n,a,b in zip(JOINTS,requested_vector,observation['observation.state'])):
            self.teleop_phase='following'
        action = self.send(requested_vector,observation)
        return {**observation,**action,'frame_id':frame_id,
                'leader_read_start_ns':leader_start,'leader_read_end_ns':leader_end,
                'leader_action_requested':requested_vector,'teleoperation_phase':self.teleop_phase,
                'tracking_error':[a-b for a,b in zip(requested_vector,observation['observation.state'])]}

    def control_sample(self):
        return self.sample(0,require_images=False)

    def close(self):
        if self.closed: return list(self._close_errors)
        self.closed = True
        errors = []
        for role,bus in reversed(list(self.buses.items())):
            try:
                if bus.is_connected:
                    # The SDK disable_torque() stops at its first failed motor. A partially
                    # enabled bus requires one independent best-effort attempt for every motor.
                    if role in self._torque_touched or self.started:
                        for motor in bus.motors:
                            for register in ('Torque_Enable','Lock'):
                                try: bus.write(register,motor,0,normalize=False,num_retry=0)
                                except Exception as exc:
                                    errors.append(f'{role}/{motor} {register} disable: {type(exc).__name__}: {exc}')
            except Exception as exc: errors.append(f'{role} disconnect: {exc}')
            finally:
                try:
                    if bus.is_connected: bus.disconnect(disable_torque=False)
                except Exception as exc: errors.append(f'{role} port close: {type(exc).__name__}: {exc}')
        if getattr(self,'camera_borrow_token',None) is not None:
            try:self.camera_provider.release_recording(self.camera_borrow_token)
            except Exception as exc:errors.append(f'camera provider release: {exc}')
            self.camera_borrow_token=None
            self.cameras={}
        for camera in self.cameras.values():
            try: camera.close()
            except Exception as exc: errors.append(f'camera close: {exc}')
        camera_lease=getattr(self,'camera_lease',None)
        if camera_lease:
            alive=any(getattr(camera,'thread',None) and camera.thread.is_alive() for camera in self.cameras.values())
            if alive:errors.append('camera reader still alive; camera lease retained')
            else:
                try:camera_lease.__exit__(None,None,None)
                except Exception as exc:errors.append(f'camera lease close: {exc}')
                self.camera_lease=None
        self.started = False
        if self.lease:
            try: self.lease.__exit__(None,None,None)
            except Exception as exc: errors.append(f'hardware lease close: {exc}')
            finally: self.lease = None
        self._close_errors = errors
        if errors:
            try:
                directory = ROOT/'logs'; directory.mkdir(exist_ok=True)
                with (directory/'hardware-stop-errors.jsonl').open('a') as out:
                    out.write(json.dumps({'timestamp_ns':time.monotonic_ns(),'errors':errors},ensure_ascii=False)+'\n')
            except OSError as exc: errors.append(f'stop error log unavailable: {exc}')
        return list(errors)
