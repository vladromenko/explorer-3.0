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
    # A servo resting at a mechanical limit reads a fraction of a degree past it (seen: elbow -0.9,
    # gripper 29.7). Within this margin the reading is pulled onto the limit so it can seed commands.
    LIMIT_MARGIN_DEG=3.

    def measured_arm(self,max_age=.25):
        measured=self.runtime.core.measured_arm(max_age)
        if measured is None:return None
        from cubebot.arm_model import SERVO_LIMITS
        values=[]
        for v,(lo,hi) in zip(measured,SERVO_LIMITS):
            if lo-self.LIMIT_MARGIN_DEG<=v<lo:v=lo
            elif hi<v<=hi+self.LIMIT_MARGIN_DEG:v=hi
            # Match the original ArmJoints ROS message conversion used by Console 8090.
            values.append(float(round(v)))
        return values
    def battery(self):return self.runtime.voltage()
    def _publish_twist(self,*values):
        with self.runtime.core_lock:self.runtime.core.on_cmd_vel(*values)
    def _publish_arm(self,servo,duration_ms):
        with self.runtime.core_lock:self.runtime.core.on_arm(servo,duration_ms)
        self.runtime.last_arm_command=list(servo);self.runtime.last_arm_at=time.monotonic()


class Runtime:
    def __init__(self, root, port='/dev/ttyUSB0'):
        self.root=Path(root);self.port=port
        self.lock=threading.RLock();self.done=threading.Event()
        self.core_lock=threading.RLock();self.transport_thread=None
        self.link=None;self.core=None;self.thread=None
        self.lease=None;self.sequence=-1;self.profile=None;self.preparing=False
        self.error=None;self.reason='Explicit enable required';self.logs=[]
        self.last_arm_command=None;self.last_arm_at=None
        self.steps=0;self.direct=None;self.direct_source=None;self.policy_session=False
        self.battery_value=None;self.battery_at=0.;self.odom=None;self.odom_at=0.
        self.scans={};self.controller_started=False
        self.session=None
        from collections import deque
        self.trace=deque(maxlen=60)  # last ~3 s of arm command/measurement/input for diagnostics
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
            self.load_mirror_overrides()
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
    # Like the native bridge, a controller arm cancellation is recovered transparently; only a
    # burst (more than ARM_CANCEL_MAX within ARM_CANCEL_WINDOW_S) ends the session.
    ARM_CANCEL_MAX=5
    ARM_CANCEL_WINDOW_S=10.
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
        """A fault in one control step ends the session, not the agent; serial loss stays fatal."""
        while not self.done.is_set():
            try:
                self.control_loop();return
            except Exception as exc:
                message=f'{type(exc).__name__}: {exc}'
                fatal=not (self.link and self.link.thread.is_alive())
                import traceback
                self.record({'at':time.time(),'event':'loop_fault','fatal':fatal,'error':message,
                             'traceback':traceback.format_exc()[-2000:]})
                with self.lock:
                    if fatal:self.error=message
                    try:self.stop(('Связь с контроллером потеряна: ' if fatal else 'Ошибка цикла управления: ')+message)
                    finally:self.log(('Фатально: ' if fatal else 'Ошибка цикла: ')+message)
                if fatal:return
                self.done.wait(.1)

    def control_loop(self):
        last=time.monotonic()
        while not self.done.wait(max(0,.05-(time.monotonic()-last))):
            now=time.monotonic();dt=now-last;last=now
            self.steps+=1
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
                    self.trace.append((round(now,3),self.last_arm_command,
                        [round(v,1) for v in self.core.servo_deg] if getattr(self.core,'servo_deg',None) is not None else None,
                        self.controls.input.get('arm') if hasattr(self.controls,'input') else None,
                        bool(self.core.state.get('arm_enabled')),bool(self.core.state.get('arm_cancel_pending')),self.core.arm_not_ready))
                    if not reason and self.arm_recovering(now,dt):pass
                    elif reason:self.stop(reason)
                    elif self.preparing:self.prepare_step(now,dt)
                    else:
                        try:
                            if self.direct_source:self.direct_step(dt,now)
                            else:self.controls.tick(dt,now)
                        except ValueError as exc:
                            # An arm target outside limits / colliding: block the arm, keep the session.
                            self.controls.jog.blocked=str(exc)[:160];self.controls.jog.velocity[:]=0
                            self.controls._arm_tracking_blocked=True
                            self.log('Рука: команда отклонена ('+str(exc)[:120]+'); отпустите клавиши')
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

    MIRROR_OVERRIDES='config/mirror-explorer.json'

    def load_mirror_overrides(self):
        path=self.root/self.MIRROR_OVERRIDES
        if path.exists():
            value=json.loads(path.read_text())
            self.controls.configure_mirror(value['sign'],value.get('gripper_map'))

    def set_mirror_config(self,sign,gripper_map=None):
        if type(sign) is not list or len(sign)!=5 or any(v not in (1,-1) for v in sign):raise ValueError('Пять знаков ±1')
        if gripper_map is not None:
            if type(gripper_map) is not list or len(gripper_map)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in gripper_map):
                raise ValueError('Две точки захвата Leader')
            if abs(gripper_map[1]-gripper_map[0])<10:raise ValueError('Откройте и закройте захват Leader полностью перед снятием точек')
        with self.lock:
            self.controls.configure_mirror(sign,gripper_map)
            path=self.root/self.MIRROR_OVERRIDES;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'sign':sign,'gripper_map':gripper_map,'saved_at':time.time(),
                                        'original':'config/mirror-original.json'},indent=2))
            self.log('Зеркало: знаки '+str(sign)+(' · захват '+str([round(v,1) for v in gripper_map]) if gripper_map else ''))
        return {'sign':sign,'gripper_map':gripper_map}

    def arm_cancel_only(self):
        core=self.core
        return (core.phase=='ready' and core.last_status is not None and core.clock()-core.last_status<.25
                and core.state.get('mode')==1 and not core.state.get('fault') and bool(core.state.get('arm_cancel_pending')))

    def arm_recovering(self,now,dt=.05):
        """True while a controller arm cancellation is being resolved; stops the session if it cannot be."""
        session=self.session
        if session is None:return False
        pending=bool(self.core.state.get('arm_cancel_pending'))
        if pending:
            if session.get('recovery_at') is None:
                session['recoveries']=session.get('recoveries',0)+1
                session['recovery_at']=now
                recent=[t for t in session.get('cancel_times',[]) if now-t<self.ARM_CANCEL_WINDOW_S]+[now]
                session['cancel_times']=recent
                self.record({'at':time.time(),'event':'arm_cancel','recovery':session['recoveries'],
                    'controller':self.controller_detail(now),'ages':self.ages(now),
                    'diagnostics':(self.core.state or {}).get('diagnostics'),'arm_command':self.last_arm_command,
                    'arm_measured':self.core.measured_arm(self.ARM_FEEDBACK_HOLD_S),'preparing':self.preparing,
                    'trace':list(self.trace)})
                if len(recent)>self.ARM_CANCEL_MAX:
                    self.stop(f'Контроллер отменил движение руки {len(recent)} раз за {self.ARM_CANCEL_WINDOW_S:.0f} с: '+self.controller_detail(now));return True
                self.log(f'Рука: контроллер отменил движение (№{session["recoveries"]}), восстановление')
                with self.core_lock:self.core.arm_target=None;self.core.arm_dirty=False
                self.controls.jog.velocity[:]=0
            # The cancellation concerns only the arm: keep the base control cycle (deadman, guard, shaping) running.
            self.io.control_step(dt)
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

    def enable(self, profile, arm_mode='cartesian', observing=False, source='human'):
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
                elif not self.model.within_limits(measured):
                    raise ValueError('Рука вне рабочих пределов '+json.dumps([round(v) for v in measured])+': сначала «Поднять руку» под наблюдением')
            with self.core_lock:self.core.enable()
            try:
                self.profile=profile;self.sequence=-1;self.lease=secrets.token_urlsafe(24)
                self.reason=None;self.preparing=profile=='prepare'
                if source not in {'human','policy'}:raise ValueError('Unknown command source')
                if source=='policy' and profile not in {'rosmaster_dock','rosmaster_pick_place'}:
                    raise ValueError('Policy/replay commands need a single-task profile')
                self.direct_source=source if source=='policy' else None;self.direct=None
                self.policy_session=source=='policy'
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

    def command(self, lease, sequence, intent, leader=None, mirror=False, direct=None, source=None):
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
                if any(abs(v)>1e-3 for v in [*intent.get('drive',[]),*intent.get('arm',[]),intent.get('pitch',0),intent.get('roll',0),intent.get('grip',0)]):
                    session['active_inputs']=session.get('active_inputs',0)+1
            if source is not None and self.policy_session:self.switch_source(source)
            if direct is not None:
                if not self.direct_source:raise PermissionError('Session is not a policy/replay session')
                self.direct=(time.monotonic(),self.validate_direct(direct))
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

    # --- policy / replay commands -------------------------------------------------------
    # Units are the recorded action units: base m/s, rad/s; arm native servo degrees. They pass
    # through the same native limits as manual control: base speed caps, lidar guard and
    # shaping; arm joint limits, path collision test, 60 deg/s and the 12 deg tracking bound.
    DIRECT_HOLD_S=.25
    DIRECT_ARM_DEG_S=60.

    def switch_source(self,source):
        """Within a policy session: hand control to the human for a correction and back."""
        if source not in {'human','policy'}:raise ValueError('Unknown command source')
        current='policy' if self.direct_source else 'human'
        if source==current:return
        self.direct=None;self.io.stop_base()
        measured=self.io.measured_arm()
        if measured is not None:self.restart_arm_from(measured)
        self.controls.input.update(drive=[0,0,0],arm=[0,0,0],pitch=0,roll=0,grip=0)
        self.direct_source='policy' if source=='policy' else None
        self.log('Источник команд: '+('политика' if source=='policy' else 'человек (коррекция)'))

    def validate_direct(self,direct):
        if self.profile=='rosmaster_dock':
            values=direct.get('base') if isinstance(direct,dict) else None
            if direct.get('arm') is not None or not isinstance(values,list) or len(values)!=3:raise ValueError('Dock action is [vx,vy,wz]')
        else:
            values=direct.get('arm') if isinstance(direct,dict) else None
            if direct.get('base') is not None or not isinstance(values,list) or len(values)!=6:raise ValueError('Arm action is six servo degrees')
        values=[float(v) for v in values]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite direct action')
        return values

    def direct_step(self,dt,now):
        if self.direct is None or now-self.direct[0]>self.DIRECT_HOLD_S:
            self.io.drive(0.,0.,0.,manual=True)
            return
        values=self.direct[1];b=self.cfg['manual_base']
        if self.profile=='rosmaster_dock':
            clip=lambda v,m:max(-m,min(m,v))
            self.io.drive(clip(values[0],b['max_vx']),clip(values[1],b['max_vy']),clip(values[2],b['max_wz']),manual=True)
            return
        self.io.drive(0.,0.,0.,manual=True)
        measured=self.io.measured_arm()
        if measured is None:raise RuntimeError('Policy arm command without fresh feedback')
        current=self.io.arm_command()
        if max(abs(a-m) for a,m in zip(current[:5],measured[:5]))>=12.:
            raise RuntimeError('Policy arm tracking error over 12 degrees')
        step=self.DIRECT_ARM_DEG_S*min(dt,.1)
        target=[c+max(-step,min(step,v-c)) for c,v in zip(current,values)]
        if max(abs(a-c) for a,c in zip(target,current))>=.5:
            self.io.move_arm(target,duration_ms=160,wait=False,check=True)

    def sample_state(self):
        """Atomic snapshot for recording: what was measured and what was sent, with agent times."""
        with self.lock:
            now=time.monotonic();core=self.core
            ns=lambda t:int(t*1e9) if t is not None else None
            measured=core.measured_arm(self.ARM_FEEDBACK_HOLD_S)
            odom_age=now-self.odom_at if self.odom_at else None
            battery_age=now-self.battery_at if self.battery_at else None
            return {'agent_monotonic_ns':time.monotonic_ns(),'control_step':self.steps,'armed':self.armed,
                'profile':self.profile,'preparing':self.preparing,'mode':self.controls.mode,
                'command_source':self.direct_source or 'human',
                'operator_intent':{k:(list(v) if isinstance(v,list) else v) for k,v in self.controls.input.items()},
                'direct_requested':self.direct[1] if self.direct else None,'direct_received_ns':ns(self.direct[0]) if self.direct else None,
                'base_requested':list(getattr(self.io,'_desired',(0.,0.,0.))),'base_after_limits':list(self.io.last_output),
                'base_guard':self.io.guard.last_block,'base_sent':list(core.cmd),'base_sent_ns':ns(core.cmd_time) if core.cmd_time>0 else None,
                'arm_sent':self.last_arm_command,'arm_sent_ns':ns(self.last_arm_at),'arm_blocked':self.controls.jog.blocked,
                # Target in force at the controller: the measured pose at enable, then each sent command.
                'arm_target':[float(v) for v in self.io.arm_command()],
                'arm_measured':measured,'arm_measured_ages_s':[now-t if t is not None else None for t in core.joint_times],
                'arm_measured_valid':list(core.joint_valid),
                'base_measured':list(self.odom) if self.odom is not None else None,'base_measured_age_s':odom_age,
                'battery_voltage':self.battery_value,'battery_age_s':battery_age,
                'controller_ready':core.ready(),'arm_cancel_recoveries':(self.session or {}).get('recoveries',0),
                'motion_commands_sent':core.motion_commands_sent,'clock_resyncs':core.clock_resyncs}

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
        self.last_arm_command=list(command);self.last_arm_at=now

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
                        'arm_recoveries':session.get('recoveries',0),'active_inputs':session.get('active_inputs',0),'arm_command':self.last_arm_command,
                        'motion_commands_sent':self.core.motion_commands_sent if self.core else None})
                    self.log('STOP: '+reason)
            self.preparing=False;self.lease=None;self.direct=None;self.direct_source=None;self.policy_session=False
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
                'mirror_sign':list(self.controls.mirror.m['sign']),'mirror_gripper_map':self.controls.gripper_map,
                'battery_voltage':self.voltage(self.BATTERY_HOLD_S if self.armed else 2.),
                'battery_age_s':round(now-self.battery_at,2) if self.battery_at else None,'measured_velocity':self.odom if now-self.odom_at<.5 else None,
                'base_command':list(self.io.last_output),'lidar_available':self.io.scan is not None,
                'motion_commands_sent':self.core.motion_commands_sent,
                'clock_sync':{'resyncs':self.core.clock_resyncs,'last_step_us':self.core.clock_last_step_us,
                              'max_step_us':self.core.clock_max_step_us},'arm_not_ready':self.core.arm_not_ready,
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
