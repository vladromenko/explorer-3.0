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
        self.session=None
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

    def voltage(self,max_age=2.):
        return self.battery_value if time.monotonic()-self.battery_at<max_age else None

    # Native firmware can freeze/omit ADC battery reports while the base streams (see
    # CubeBot battery.py); a fresh value is required to enable, the last one is kept while armed.
    BATTERY_HOLD_S=30.
    LOOP_DEADLINE_S=.3
    ARM_FEEDBACK_HOLD_S=.6
    # Poses are streamed as short targets (like the native jog) instead of one long native
    # trajectory resent every tick; the controller cancelled those during preparation.
    POSE_DEG_S=20.
    POSE_STEP_MS=160
    POSE_MAX_LEAD_DEG=10.
    # Controller-side arm cancellation (stale servo feedback / expired arm lease) is recoverable
    # a few times per session; the base is held and arm targets restart from measured angles.
    ARM_RECOVERIES=3
    ARM_RECOVERY_S=1.5

    def ages(self,now=None):
        now=time.monotonic() if now is None else now
        core=self.core
        return {'battery_s':round(now-self.battery_at,3) if self.battery_at else None,
                'controller_status_s':round(now-core.last_status,3) if core and core.last_status is not None else None,
                'joints_s':[round(now-t,3) if t is not None else None for t in core.joint_times] if core else None,
                'joints_valid':list(core.joint_valid) if core else None,
                'operator_left_s':round(core.operator_until-now,3) if core else None}

    def record(self,event):
        try:
            path=self.root/'logs/control-sessions.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as output:output.write(json.dumps(event,ensure_ascii=False,default=str)+'\n')
        except OSError:pass

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
                        reason=None
                        volts=self.voltage(self.BATTERY_HOLD_S)
                        if self.session:self.session['max_loop_dt']=max(self.session['max_loop_dt'],dt)
                        if dt>self.LOOP_DEADLINE_S:reason=f'Control loop deadline missed ({dt:.3f} s)'
                        elif not self.core.ready() and not self.arm_cancel_only():
                            reason='Controller feedback lost: '+self.controller_detail(now)
                        elif volts is None:reason=f'Battery telemetry stale (>{self.BATTERY_HOLD_S:.0f} s)'
                        elif volts<self.cfg['mission']['min_battery_v']:reason='Low battery'
                        elif (self.root/'STOP').exists():reason='STOP file'
                        elif now>=self.core.operator_until:reason='Browser / Homebrain heartbeat lost'
                        elif self.profile!='rosmaster_dock' and self.core.measured_arm(self.ARM_FEEDBACK_HOLD_S) is None:
                            reason='Arm feedback lost: '+json.dumps(self.ages(now)['joints_s'])
                        if not reason and self.arm_recovering(now):pass
                        elif reason:self.stop(reason)
                        elif self.preparing:self.prepare_step(now,dt)
                        else:
                            self.controls.tick(dt,now)
                            if not self.controls.armed:self.stop('Native controls stopped: '+str(self.controls.stop_reason or self.controls.mirror.blocked or self.controls.jog.blocked))
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

    def arm_cancel_only(self):
        core=self.core
        return (core.phase=='ready' and core.last_status is not None and core.clock()-core.last_status<.25
                and core.state.get('mode')==1 and not core.state.get('fault') and bool(core.state.get('arm_cancel_pending')))

    def arm_recovering(self,now):
        """True while a controller arm cancellation is being resolved; stops the session if it cannot be."""
        session=self.session
        if session is None:return False
        pending=bool(self.core.state.get('arm_cancel_pending'))
        if pending:
            if session.get('recovery_at') is None:
                session['recoveries']=session.get('recoveries',0)+1
                session['recovery_at']=now
                self.record({'at':time.time(),'event':'arm_cancel','recovery':session['recoveries'],
                    'controller':self.controller_detail(now),'ages':self.ages(now),
                    'diagnostics':(self.core.state or {}).get('diagnostics'),'arm_command':self.last_arm_command,
                    'arm_measured':self.core.measured_arm(self.ARM_FEEDBACK_HOLD_S),'preparing':self.preparing})
                if session['recoveries']>self.ARM_RECOVERIES:
                    self.stop(f'Контроллер отменил движение руки {session["recoveries"]} раз: '+self.controller_detail(now));return True
                self.log(f'Рука: контроллер отменил движение ({session["recoveries"]}/{self.ARM_RECOVERIES}), восстановление')
                with self.core_lock:self.core.arm_target=None;self.core.arm_dirty=False
                self.controls.jog.velocity[:]=0
            self.io.stop_base()
            if now-session['recovery_at']>self.ARM_RECOVERY_S:
                self.stop('Отмена движения руки не снялась: '+self.controller_detail(now));return True
            return True
        if session.get('recovery_at') is not None:
            measured=self.core.measured_arm(self.ARM_FEEDBACK_HOLD_S)
            if measured is None:
                if now-session['recovery_at']>self.ARM_RECOVERY_S:self.stop('Нет обратной связи руки после отмены')
                return True
            session['recovery_at']=None
            self.restart_arm_from(measured)
            self.log('Рука: восстановлено от измеренной позы')
            return False
        disabled=self.profile!='rosmaster_dock' and self.core.arm_target is not None and not self.core.state.get('arm_enabled')
        if disabled:
            since=session.setdefault('arm_disabled_at',now)
            if now-since>self.ARM_RECOVERY_S:
                self.stop('Контроллер не включил руку: '+self.controller_detail(now));return True
        else:session.pop('arm_disabled_at',None)
        return False

    def restart_arm_from(self,measured):
        """New arm targets start from where the arm really is; no old target is replayed."""
        self.io._arm_cmd=list(measured)
        self.controls.jog.reset(measured);self.controls.jog.velocity[:]=0
        self.controls._arm_tracking_blocked=False;self.controls._arm_input_active=False
        if self.controls.mode=='mirror':
            self.controls.mirror.servo=np.array(measured,float);self.controls.mirror.target=self.controls.mirror.servo.copy()
            self.controls.mirror.rebase()
        if self.preparing:self.pose_cmd=list(measured)

    def controller_detail(self,now):
        core=self.core;state=core.state or {}
        age=now-core.last_status if core.last_status is not None else None
        return (f"phase={core.phase} status_age={age if age is None else round(age,3)} mode={state.get('mode')} "
                f"fault={state.get('fault')} cancel_pending={state.get('arm_cancel_pending')}")

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
                    self.begin_pose([90]*6,measured,time.monotonic())
                else:
                    self.controls.arm_mode=arm_mode
                    self.controls.enable(profile,time.monotonic())
                self.session={'started':time.time(),'started_monotonic':time.monotonic(),'profile':profile,
                    'arm_mode':arm_mode,'commands':0,'stale_commands':0,'max_command_gap':0.,
                    'last_command':time.monotonic(),'max_loop_dt':0.,'battery_at_enable':volts}
                self.log('Управление включено: '+profile)
                return {'lease':self.lease,'profile':profile}
            except Exception:
                self.stop('Enable failed')
                raise

    def command(self, lease, sequence, intent, leader=None, mirror=False):
        with self.lock:
            if not self.lease or not secrets.compare_digest(lease,self.lease):raise PermissionError('Сессия остановлена')
            if type(sequence) is not int:raise ValueError('Invalid command sequence')
            if sequence<=self.sequence:
                # A request delayed past a client-side timeout: drop it; the newer one already applied.
                if self.session:self.session['stale_commands']+=1
                return self.status()
            with self.core_lock:self.core.heartbeat()
            if self.session:
                now=time.monotonic();session=self.session
                session['max_command_gap']=max(session['max_command_gap'],now-session['last_command'])
                session['last_command']=now;session['commands']+=1
            if self.controls.armed:
                # Keep operator input fresh during a preset too; it is applied only after the pose.
                self.controls.set_input(intent if not self.preparing else {**intent,'arm':[0,0,0],'pitch':0,'roll':0,'grip':0},time.monotonic())
                if self.profile=='rosmaster_manual':self.io.guard_off=bool(intent.get('free',False))
                if leader is not None and not self.controls.leader(leader,time.monotonic()):raise ValueError('Invalid Leader frame')
                if mirror and self.controls.mode!='mirror':self.controls.enable_mirror(time.monotonic())
                elif not mirror and self.controls.mode=='mirror':
                    measured=self.io.measured_arm()
                    self.controls.mode='manual';self.io.stop_base()
                    if measured is not None:self.restart_arm_from(measured)
                    self.log('Зеркало выключено; ручное управление продолжается')
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
            target=[float(v) for v in self.cfg['arm'][poses[name]]]
            if not self.model.within_limits(target):raise ValueError('Поза вне пределов руки')
            start=np.array(measured,float);end=np.array(target,float)
            for frac in (0.,.25,.5,.75,1.):
                if not self.model.collision_free(start+(end-start)*frac,self.cfg['arm']['floor_margin_m']):
                    raise ValueError('Путь к позе пересекает пол или шасси')
            self.io._arm_cmd=list(measured)
            self.io.stop_base()
            self.controls.jog.velocity[:]=0
            self.begin_pose(target,measured,time.monotonic())
            self.log('arm -> '+name)

    def begin_pose(self,target,measured,now):
        self.pose_target=[float(v) for v in target];self.pose_cmd=[float(v) for v in measured]
        distance=max(abs(a-b) for a,b in zip(self.pose_target,self.pose_cmd))
        self.prepare_started=now;self.pose_deadline=now+distance/self.POSE_DEG_S+4.
        self.preparing=True

    def prepare_step(self,now,dt=.05):
        measured=self.core.measured_arm(self.ARM_FEEDBACK_HOLD_S)
        if measured is None:self.stop('Feedback lost during arm preparation');return
        error=[abs(v-target) for v,target in zip(measured,self.pose_target)]
        if max(error[:5])<4 and error[5]<8:
            self.io._arm_cmd=list(measured);self.controls.jog.reset(measured)
            if self.profile=='prepare' or not self.controls.armed:
                self.stop('Поза достигнута: подтверждено измерениями');return
            self.preparing=False;self.controls.last_input=now
            self.log('Поза достигнута; ручное управление продолжается');return
        if now>self.pose_deadline:
            self.stop('Поза не достигнута за отведённое время: '+json.dumps([round(v,1) for v in measured]));return
        # Advance the commanded pose at a limited rate, but never far ahead of the measured arm.
        lead=max(abs(c-m) for c,m in zip(self.pose_cmd,measured))
        if lead<self.POSE_MAX_LEAD_DEG:
            limit=self.POSE_DEG_S*min(dt,.1)
            self.pose_cmd=[c+max(-limit,min(limit,t-c)) for c,t in zip(self.pose_cmd,self.pose_target)]
        command=[float(round(v)) for v in self.pose_cmd]
        with self.core_lock:self.core.on_arm(command,self.POSE_STEP_MS)
        self.last_arm_command=list(command)

    def stop(self,reason='Operator STOP'):
        # Cancel transport before waiting for an IK calculation holding the control lock.
        if self.core:
            with self.core_lock:
                if not self.core.permitted and self.core.stop_reason and self.session is not None:
                    reason=self.core.stop_reason+' (затем: '+reason+')'
                self.core.disarm(reason)
        with self.lock:
            # Keep the first cause: a later operator STOP must not hide why the session ended.
            session,self.session=self.session,None
            if session is not None or self.lease is not None or self.reason is None:
                self.reason=reason
                if session is not None:
                    now=time.monotonic()
                    self.record({'at':time.time(),'event':'stop','reason':reason,'profile':session['profile'],
                        'arm_mode':session['arm_mode'],'duration_s':round(now-session['started_monotonic'],3),
                        'commands':session['commands'],'stale_commands':session['stale_commands'],
                        'max_command_gap_s':round(session['max_command_gap'],3),
                        'since_last_command_s':round(now-session['last_command'],3),
                        'max_loop_dt_s':round(session['max_loop_dt'],3),'ages':self.ages(now),
                        'battery_v':self.battery_value,'battery_at_enable':session['battery_at_enable'],
                        'controller':self.controller_detail(now) if self.core else None,
                        'controller_diagnostics':(self.core.state or {}).get('diagnostics') if self.core else None,
                        'arm_recoveries':session.get('recoveries',0),'arm_command':self.last_arm_command,
                        'motion_commands_sent':self.core.motion_commands_sent if self.core else None})
                    self.log('STOP: '+reason)
            self.preparing=False;self.lease=None
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
                'battery_voltage':self.voltage(self.BATTERY_HOLD_S if self.armed else 2.),
                'battery_age_s':round(now-self.battery_at,2) if self.battery_at else None,'measured_velocity':self.odom if now-self.odom_at<.5 else None,
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
