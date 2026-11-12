"""One serial owner using copied native transport and Console controls, without ROS commands."""
import fcntl
import json
import math
import secrets
import termios
import threading
import time
from pathlib import Path
import numpy as np
from cubebot import config
from cubebot.arm_model import ArmModel
from cubebot.robot_io import BaseIO
from cubebot.mirror import load as mirror_config
from cubebot.scan_matching import scan_to_base, build_scan
from rosmaster.link import Link
from leased_core import LeasedCore
from control_adapter import NativeControls


class NativeIO(BaseIO):
    REALTIME=True

    def __init__(self, runtime, cfg, model):
        super().__init__(cfg,model,runtime.log)
        self.runtime=runtime

    def now(self):return time.monotonic()
    def measured_arm(self,max_age=.25):
        measured=self.runtime.core.measured_arm(max_age)
        # Match the original ArmJoints ROS message conversion used by Console 8090.
        return [float(round(v)) for v in measured] if measured is not None else None
    def battery(self):return self.runtime.voltage()
    def _publish_twist(self,*values):
        with self.runtime.core_lock:self.runtime.core.on_cmd_vel(*values)
    def _publish_arm(self,servo,duration_ms):
        with self.runtime.core_lock:self.runtime.core.on_arm(servo,duration_ms)
        self.runtime.last_arm_command=list(servo)


class Runtime:
    def __init__(self, root, port='/dev/ttyUSB0'):
        self.root=Path(root);self.port=port
        self.lock=threading.RLock();self.done=threading.Event()
        self.core_lock=threading.RLock();self.transport_thread=None
        self.link=None;self.core=None;self.thread=None
        self.lease=None;self.sequence=-1;self.profile=None;self.preparing=False
        self.error=None;self.reason='Explicit enable required';self.logs=[]
        self.last_arm_command=None
        self.battery_value=None;self.battery_at=0.;self.odom=None;self.odom_at=0.
        self.scans={};self.controller_started=False
        self.cfg=config.load()
        self.cfg['mission']['stop_file']=str(self.root/'STOP')
        self.model=ArmModel(self.cfg['arm']['tcp_offset_m'])

    @property
    def armed(self):return bool(self.core and self.core.permitted)

    def log(self,message):
        self.logs=(self.logs+[str(message)])[-20:]

    def start(self):
        calibration=json.loads((self.root/'config/controller-calibration.json').read_text())['joints']
        self.link=Link(self.port)
        try:
            self.link.port.write_timeout=.1
            fcntl.ioctl(self.link.port.fileno(),termios.TIOCEXCL)
            self.core=LeasedCore(self.link,calibration,self.receive,log=self.log)
            self.link.callbacks.append(self.core.on_frame)
            self.io=NativeIO(self,self.cfg,self.model)
            self.controls=NativeControls(self.io,self.cfg,self.model,
                mirror_config(self.root/'config/mirror-original.json'),self.log)
            self.transport_thread=threading.Thread(target=self.transport_loop,daemon=True)
            self.transport_thread.start()
            self.thread=threading.Thread(target=self.run,daemon=True)
            self.thread.start()
        except Exception:
            self.link.close();self.link=None
            raise

    def voltage(self):
        return self.battery_value if time.monotonic()-self.battery_at<2 else None

    def receive(self,topic,data):
        now=time.monotonic()
        if topic=='battery':self.battery_value=data;self.battery_at=now
        elif topic=='odom':self.odom=data['velocity'];self.odom_at=now
        elif topic.startswith('scan'):
            self.scans[topic]=(data,now)

    def run(self):
        last=time.monotonic()
        try:
            while not self.done.wait(max(0,.05-(time.monotonic()-last))):
                now=time.monotonic();dt=now-last;last=now
                with self.lock:
                    if self.armed:
                        volts=self.voltage()
                        reason=None
                        if dt>.15:reason='Control loop deadline missed'
                        elif not self.core.ready():reason='Controller feedback lost or cancellation pending'
                        elif volts is None:reason='Battery telemetry stale'
                        elif volts<self.cfg['mission']['min_battery_v']:reason='Low battery'
                        elif (self.root/'STOP').exists():reason='STOP file'
                        elif now>=self.core.operator_until:reason='Browser / Homebrain heartbeat lost'
                        elif self.profile!='rosmaster_dock' and self.core.measured_arm() is None:reason='Arm feedback lost'
                        if reason:self.stop(reason)
                        elif self.preparing:self.prepare_step(now)
                        else:
                            self.controls.tick(dt,now)
                            if not self.controls.armed:self.stop('Native controls stopped: '+str(self.controls.mirror.blocked or self.controls.jog.blocked))
                            else:
                                parts=[]
                                for name,(scan,stamp) in self.scans.items():
                                    if now-stamp<.5:
                                        parts.append(scan_to_base(scan['ranges'],0,2*math.pi/666,.02,12,
                                            self.cfg['odometry']['lidar_mounts'][name]))
                                self.io.scan=build_scan(parts,self.cfg['odometry']['lidar_self_mask']) if parts else None
                                self.io.control_step(dt)
                    elif self.lease is not None:
                        self.stop(self.core.stop_reason or 'Controller stopped')
                    if not self.link.thread.is_alive():raise RuntimeError('Serial reader exited')
        except Exception as exc:
            with self.lock:
                self.error=f'{type(exc).__name__}: {exc}'
                try:self.stop(self.error)
                finally:self.log(self.error)

    def transport_loop(self):
        try:
            while not self.done.wait(.02):
                with self.core_lock:self.core.tick()
        except Exception as exc:
            self.error='Native transport: '+str(exc)
            with self.core_lock:self.core.disarm(self.error)

    def enable(self, profile, arm_mode='cartesian', observing=False):
        with self.lock:
            if self.armed:raise ValueError('Сначала STOP')
            if self.error:raise RuntimeError(self.error)
            if self.done.is_set():raise RuntimeError('Runtime closed')
            if observing is not True:raise ValueError('Подтвердите свободное пространство и присутствие рядом с роботом')
            if profile not in {'rosmaster_dock','rosmaster_pick_place','rosmaster_manual','prepare'}:raise ValueError('Unknown profile')
            if arm_mode not in {'cartesian','joint'}:raise ValueError('Unknown arm mode')
            volts=self.voltage()
            if volts is None or volts<self.cfg['mission']['min_battery_v']:raise ValueError('Нет свежего допустимого напряжения батареи')
            if profile!='rosmaster_dock':
                measured=self.io.measured_arm()
                if measured is None:raise ValueError('Нужны свежие измерения всех шести приводов')
                if self.odom is None or time.monotonic()-self.odom_at>.25 or any(abs(v)>.01 for v in self.odom):
                    raise ValueError('Шасси должно быть неподвижно по свежей телеметрии')
                if profile=='prepare':
                    if observing is not True:raise ValueError('Подтвердите свободное пространство и наблюдение за рукой')
                    if not self.model.collision_free([90]*6,self.cfg['arm']['floor_margin_m']):raise ValueError('Исходная поза не проходит прежнюю модель')
                elif not self.model.within_limits(measured):raise ValueError('Рука сложена: сначала «Подготовить руку» под наблюдением')
            with self.core_lock:self.core.enable()
            try:
                self.profile=profile;self.sequence=-1;self.lease=secrets.token_urlsafe(24)
                self.reason=None;self.preparing=profile=='prepare'
                if self.preparing:
                    self.prepare_started=time.monotonic();self.pose_target=[90]*6;self.pose_runtime=4000
                else:
                    self.controls.arm_mode=arm_mode
                    self.controls.enable(profile,time.monotonic())
                return {'lease':self.lease,'profile':profile}
            except Exception:
                self.stop('Enable failed')
                raise

    def command(self, lease, sequence, intent, leader=None, mirror=False):
        with self.lock:
            if not self.lease or not secrets.compare_digest(lease,self.lease):raise PermissionError('Сессия остановлена')
            if type(sequence) is not int or sequence<=self.sequence:raise ValueError('Out-of-order command')
            with self.core_lock:self.core.heartbeat()
            if not self.preparing:
                self.controls.set_input(intent,time.monotonic())
                if self.profile=='rosmaster_manual':self.io.guard_off=bool(intent.get('free',False))
                if leader is not None and not self.controls.leader(leader,time.monotonic()):raise ValueError('Invalid Leader frame')
                if mirror and self.controls.mode!='mirror':self.controls.enable_mirror(time.monotonic())
            self.sequence=sequence
            return self.status()

    def preset(self,lease,name):
        with self.lock:
            if not self.lease or not secrets.compare_digest(lease,self.lease) or not self.armed:raise PermissionError('Сначала включите управление')
            if self.profile not in {'rosmaster_manual','rosmaster_pick_place'}:raise PermissionError('Нужен профиль руки')
            poses={'look':'search_pose','near':'near_search_pose','carry':'carry_pose','home':'safe_pose'}
            if name not in poses:raise ValueError('Unknown native preset')
            if self.odom is None or time.monotonic()-self.odom_at>.25 or any(abs(v)>.01 for v in self.odom):raise ValueError('Сначала остановите шасси')
            measured=self.io.measured_arm()
            if measured is None:raise ValueError('Нет свежих измерений руки')
            self.io._arm_cmd=list(measured)
            self.io.stop_base()
            self.io.move_arm(list(self.cfg['arm'][poses[name]]),wait=False)
            self.pose_target=list(self.last_arm_command)
            self.pose_runtime=self.core.arm_target[1]
            self.prepare_started=time.monotonic();self.preparing=True
            self.log('arm -> '+name)

    def prepare_step(self,now):
        measured=self.core.measured_arm()
        if measured is None:self.stop('Feedback lost during arm preparation');return
        elapsed=now-self.prepare_started
        if elapsed>self.pose_runtime/1000 and max(abs(v-target) for v,target in zip(measured,self.pose_target))<4:
            self.stop('Поза достигнута: подтверждено измерениями');return
        if elapsed>self.pose_runtime/1000+3:self.stop('Поза не достигнута за отведённое время');return
        # The same 90-degree / 4-second preparation used by the former startup.
        # No changes to raw calibration or limits and no automatic execution at boot.
        with self.core_lock:self.core.on_arm(self.pose_target,self.pose_runtime)
        self.last_arm_command=list(self.pose_target)

    def stop(self,reason='Operator STOP'):
        # Cancel transport before waiting for an IK calculation holding the control lock.
        if self.core:
            with self.core_lock:self.core.disarm(reason)
        with self.lock:
            self.reason=reason;self.preparing=False;self.lease=None
            if self.core:
                try:self.controls.stop_all()
                finally:
                    with self.core_lock:self.core.disarm(reason)

    def status(self):
        with self.lock:
            now=time.monotonic()
            return {'armed':self.armed,'profile':self.profile,'preparing':self.preparing,
                'reason':self.reason or self.core.stop_reason,'error':self.error,
                'controller_phase':self.core.phase,'controller_ready':self.core.ready(),
                'controller':self.core.state if self.core.last_status is not None and now-self.core.last_status<.5 else None,
                'arm_measured':self.core.measured_arm(),'arm_feedback_age_s':[
                    now-t if t is not None else None for t in self.core.joint_times],
                'arm_command':self.last_arm_command,'arm_blocked':self.controls.jog.blocked,
                'arm_mode':self.controls.arm_mode,
                'tcp':[round(float(v),3) for v in self.model.tcp_position(self.io.arm_command())],
                'guard':self.io.guard.last_block,
                'mirror_target':[round(float(v),1) for v in self.controls.mirror.target],
                'mirror_blocked':self.controls.mirror.blocked,'mode':self.controls.mode,
                'battery_voltage':self.voltage(),'measured_velocity':self.odom if now-self.odom_at<.5 else None,
                'base_command':list(self.io.last_output),'lidar_available':self.io.scan is not None,
                'motion_commands_sent':self.core.motion_commands_sent,
                'physical_stop_confirmed':False,'logs':list(self.logs)}

    def close(self):
        try:self.stop('Runtime closing')
        finally:
            self.done.set()
            if self.thread:self.thread.join(timeout=2)
            if self.transport_thread:self.transport_thread.join(timeout=2)
            if self.link:
                try:self.link.estop()
                finally:self.link.close()
