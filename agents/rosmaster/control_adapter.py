"""Explorer adapter over the selected CubeBot controls; no ROS or device construction.

_manual_step and _mirror_step are verbatim copies from the rollback snapshot of
Console 8090. The wrapper validates input, separates profiles and expires a
session. Manual XYZ adapts the grasp IK objective using native geometry and
SciPy; native CartesianJog/MirrorMapper and calibration remain copied sources.
"""
import math
import numpy as np


class ManualCartesianModel:
    """Native URDF/FK and SciPy position-first jogging; native IK for pitch only.

    Native IK's 10 mm grasp tolerance and strong pitch preference are too loose
    for sub-millimetre jog steps. This only adapts manual XYZ, not mirror or poses.
    """
    def __init__(self,model):self.model=model
    def __getattr__(self,name):return getattr(self.model,name)
    def tcp_and_jacobian(self,servo):
        from cubebot.arm_model import CHAIN
        fk=self.model.fk(servo);point=fk['tcp'][:3,3]
        columns=[]
        for i,frame in enumerate(fk['frames'][:4]):
            axis=np.asarray(CHAIN[i][2],float);axis/=np.linalg.norm(axis)
            world_axis=frame[:3,:3]@axis
            columns.append(np.cross(world_axis,point-frame[:3,3])*math.pi/180*(-1 if i==0 else 1))
        return point,np.column_stack(columns)
    def ik(self,target,pitch,seed,**kwargs):
        start=self.model.tcp_position(seed)
        requested=np.asarray(target)-start;length=float(np.linalg.norm(requested))
        if length<=1e-6:return self.model.ik(target,pitch,seed,**kwargs)
        from scipy.optimize import least_squares
        from cubebot.arm_model import SERVO_LIMITS
        seed4=np.asarray(seed[:4],float);step_limit=kwargs['max_step_deg']
        lo=np.maximum(np.array(SERVO_LIMITS[:4])[:,0],seed4-step_limit)
        hi=np.minimum(np.array(SERVO_LIMITS[:4])[:,1],seed4+step_limit)
        roll=kwargs.get('roll_servo',seed[4])
        goal=np.asarray(target)
        cache=[None,None]
        def geometry(joints):
            key=tuple(joints)
            if key!=cache[0]:cache[:]=[key,self.tcp_and_jacobian([*joints,roll,seed[5]])]
            return cache[1]
        def residual(joints):
            # Millimetres keep the optimizer's gradient tolerance meaningful for
            # sub-millimetre jog requests. A small posture cost selects the local
            # solution; tool pitch is free during XYZ translation.
            return np.r_[1000*(geometry(joints)[0]-goal),.01*(joints-seed4)]
        def jacobian(joints):return np.vstack([1000*geometry(joints)[1],.01*np.eye(4)])
        for scale in (1.,.5,.25,.125):
            # Near full extension the Cartesian step may need more joint travel
            # than this tick permits. Reduce speed, never relax joint rate limits.
            goal=start+requested*scale
            fit=least_squares(residual,np.clip(seed4,lo,hi),bounds=(lo,hi),jac=jacobian,
                max_nfev=10,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            sol=np.r_[fit.x,roll]
            step=self.model.tcp_position([*sol,seed[5]])-start
            error=float(np.linalg.norm(step-requested))
            if not self.model.collision_free([*sol,seed[5]]):continue
            direction=requested/length;along=float(np.dot(step,direction))
            sideways=float(np.linalg.norm(step-along*direction))
            # Do not accumulate lateral motion over many small/shortened steps.
            if along>0 and sideways<=max(.00001,.05*along):return sol,error
        return None,error


class NativeControls:
    def __init__(self, io, cfg, arm_model, mirror_cfg, log=lambda message: None):
        from cubebot.arm_control import CartesianJog
        from cubebot.mirror import MirrorMapper
        self.io, self.cfg, self.arm, self.log = io, cfg, arm_model, log
        self.jog = CartesianJog(ManualCartesianModel(arm_model), cfg, io.arm_command())
        # The copied task box capped Z at .42 m, below the actual upright TCP
        # (.544 m). It prevented returning upward after any downward jog.
        # Native joint limits, reachability and floor/chassis geometry bound XYZ.
        self.jog.WORKSPACE=((-math.inf,math.inf),)*3
        self.mirror = MirrorMapper(arm_model, cfg['arm'], mirror_cfg, io.arm_command())
        self.original_scale = list(mirror_cfg['scale'])
        self.input = dict(drive=[0,0,0], arm=[0,0,0], pitch=0, roll=0, grip=0, slow=False)
        self.last_input = -float('inf')
        self._arm_input_active = False
        self._arm_tracking_blocked = False
        self.mode = 'manual'
        self.arm_mode = 'cartesian'
        self.profile = None
        self.armed = False
        self.stop_reason = None
        self.gripper_map = None
        self.ref_pair = None        # (leader 5 joints deg, robot servo 5 joints) captured in the same physical pose
        self.last_leader = None

    def enable(self, profile, now):
        if self.armed:raise ValueError('Stop the previous control session first')
        if profile not in {'rosmaster_dock','rosmaster_pick_place','rosmaster_manual'}:raise ValueError('Unknown profile')
        if profile in {'rosmaster_pick_place','rosmaster_manual'}:
            measured=self.io.measured_arm(max_age=.25)
            if measured is None:raise ValueError('Fresh native arm feedback required; estimates are not measurements')
            self.io._arm_cmd=list(measured)
            self.jog.reset(measured)
        self.profile=profile;self.armed=True;self.last_input=now;self.stop_reason=None

    def set_input(self, value, now):
        if not self.armed:raise PermissionError('Explicit enable required')
        x={**self.input,**{k:value[k] for k in self.input if k in value}}
        numbers=[*x['drive'],*x['arm'],x['pitch'],x['roll'],x['grip']]
        if len(x['drive'])!=3 or len(x['arm'])!=3 or not all(math.isfinite(v) and -1<=v<=1 for v in numbers):
            raise ValueError('Expected finite normalized control intents')
        if not isinstance(x['slow'],bool):raise ValueError('slow must be boolean')
        if self.profile=='rosmaster_dock' and any(x['arm']+[x['pitch'],x['roll'],x['grip']]):
            raise PermissionError('Dock profile cannot move arm')
        if self.profile=='rosmaster_pick_place' and any(x['drive']):
            raise PermissionError('Arm profile requires stationary base')
        x['drive']=list(x['drive']);x['arm']=list(x['arm'])
        self.input=x;self.last_input=now

    def set_arm_mode(self, mode):
        if mode not in {'cartesian','joint'}:raise ValueError('Unknown arm mode')
        self.stop_all()
        self.arm_mode=mode
        self.jog.reset(self.io.measured_arm(max_age=.25) or self.io.arm_command())

    JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')

    def leader(self, positions, now):
        self.last_leader = dict(positions)
        # Absolute gripper: the operator's captured Leader open/closed values map to the native
        # gripper_open..gripper_close range (the copied relative mode cannot reach full close).
        if self.gripper_map is not None:
            opened,closed=self.gripper_map
            try:value=float(positions['gripper'])
            except (KeyError,TypeError,ValueError):return False
            positions={**positions,'gripper':min(100.,max(0.,(value-opened)/(closed-opened)*100.))}
        return self.mirror.update(positions,now)

    def configure_mirror(self, sign, gripper_map, ref_pair=None):
        """Operator-checked joint directions, gripper endpoints and pose pair; only while the mirror is off."""
        if self.mode=='mirror':raise PermissionError('Сначала выключите зеркало')
        self.ref_pair = (list(ref_pair['leader']), list(ref_pair['servo'])) if ref_pair else None
        # With a captured pose pair the mirror is absolute and 1:1, so the arm can match the Leader's pose.
        self.mirror.m['scale'] = [1.0]*5 if self.ref_pair else list(self.original_scale)
        self.mirror.m['sign']=list(sign)
        self.mirror.m['relative_gripper']=gripper_map is None
        self.gripper_map=tuple(gripper_map) if gripper_map is not None else None
        self.mirror.rebase()

    def mapped_pose(self, measured):
        """Robot servo pose that corresponds to the latest Leader frame (needs the pose pair)."""
        from cubebot.arm_model import SERVO_LIMITS
        if self.ref_pair is None or self.last_leader is None:return None
        leader_ref,servo_ref=self.ref_pair;sign=self.mirror.m['sign']
        pose=[servo_ref[i]+sign[i]*(float(self.last_leader[name])-leader_ref[i]) for i,name in enumerate(self.JOINTS)]
        a=self.cfg['arm']
        if self.gripper_map is not None:
            opened,closed=self.gripper_map
            g=min(1.,max(0.,(float(self.last_leader['gripper'])-opened)/(closed-opened)))
            pose.append(a['gripper_open']+(a['gripper_close']-a['gripper_open'])*g)
        else:pose.append(measured[5])
        return [min(hi-1.,max(lo+1.,v)) for v,(lo,hi) in zip(pose,SERVO_LIMITS)]  # 1 deg inside the limits

    def enable_mirror(self, now):
        if not self.armed or self.profile not in {'rosmaster_pick_place','rosmaster_manual'}:raise PermissionError('Arm session required')
        measured=self.io.measured_arm(max_age=.25)
        if measured is None or now-self.mirror.last_msg>self.mirror.m['timeout_s']:
            raise ValueError('Fresh arm feedback and Leader stream required')
        self.mirror.servo=np.array(measured,float)
        self.mirror.target=self.mirror.servo.copy()
        self.mirror.rebase()
        self.io._arm_cmd=list(measured)
        self.mode='mirror'
        if self.ref_pair is not None and self.last_leader is not None:
            # Absolute mode: the stored pair is the reference instead of 'wherever both are now'.
            leader_ref,servo_ref=self.ref_pair
            self.mirror.ref=(list(leader_ref),np.array(servo_ref,float),float(self.last_leader['gripper']),float(measured[5]))

    # A late control loop (Python GC, camera/IK load) is not by itself a reason to end the
    # session: the native arm jog still blocks steps longer than 0.15 s, and the controller
    # lease (240 ms) and the zero-motion rule for input older than 0.4 s stay the hard limits.
    MAX_DT=.3

    def tick(self, dt, now):
        if self.armed:
            if not math.isfinite(dt) or not 0<dt<=self.MAX_DT:
                self.stop_reason=f'control step {dt:.3f} s';self.stop_all()
            elif now-self.last_input>=1.:  # motion is already zero after 0.4 s in _manual_step
                self.stop_reason=f'no operator input for {now-self.last_input:.2f} s';self.stop_all()
            elif self.mode=='mirror':
                self._mirror_step(dt,now)
                if not self.armed:
                    # The verbatim mirror step stopped everything; keep the session in manual
                    # mode with zero intent instead, so the operator can re-enable the mirror.
                    reason=self.mirror.blocked
                    self.armed=True;self.mode='manual';self.last_input=now
                    self.jog.reset(self.io.measured_arm(max_age=.25) or self.io.arm_command())
                    self.mirror.blocked=reason
                    self.log('Зеркало остановлено: '+str(reason)+'; рука стоит, ручное управление доступно')
            else:self._manual_step(dt,now)

    def stop_all(self):
        self.armed=False
        self.input.update(drive=[0,0,0],arm=[0,0,0],pitch=0,roll=0,grip=0)
        self.jog.velocity[:]=0
        self._arm_input_active=False;self._arm_tracking_blocked=False
        self.mode='manual';self.mirror.rebase()
        self.io.stop_base()
        self.log('STOP: no automatic resume; last short native arm target may finish')

    def _manual_step(self, dt, now):
        fresh = now - self.last_input < 0.4
        x = self.input
        if not fresh:
            x = dict(drive=[0, 0, 0], arm=[0, 0, 0], pitch=0, roll=0, grip=0, slow=False)
            self.jog.velocity[:] = 0
        speed = 0.4 if x["slow"] else 1.0
        b = self.cfg["manual_base"]
        self.io.drive(x["drive"][0] * b["max_vx"] * speed, x["drive"][1] * b["max_vy"] * speed,
                      x["drive"][2] * b["max_wz"] * speed, manual=True)
        active = any(abs(v) > .001 for v in [*x["arm"], x["pitch"], x["roll"], x["grip"]])
        measured = self.io.measured_arm(max_age=.25)
        if active and not self._arm_input_active:
            if measured is None:
                self.jog.blocked = "no fresh arm feedback"
            else:
                self.io._arm_cmd = list(measured)
                self.jog.reset(measured)
        tracking_ok = measured is not None and max(abs(a-b) for a,b in
            zip(self.io.arm_command()[:5], measured[:5])) < 12.
        if not active:
            self._arm_tracking_blocked = False
            self.jog.velocity[:] = 0
        if active and (not tracking_ok or not 0 < dt <= .15):
            self._arm_tracking_blocked = True
        if self._arm_tracking_blocked:
            self.jog.blocked = "arm tracking error; release controls"
            self.jog.velocity[:] = 0
        elif active:
            if self.arm_mode == "joint":
                rates = [-x["arm"][1], x["arm"][0], x["arm"][2], x["pitch"], x["roll"], x["grip"]]
                cmd = self.jog.step_joints(dt, rates, x["slow"])
            else:
                cmd = self.jog.step(dt, x["arm"], x["pitch"], x["roll"], x["grip"], x["slow"])
            if cmd:
                self.io.move_arm(cmd, duration_ms=self.jog.run_ms, wait=False, check=True)
        self._arm_input_active = active

    def _mirror_step(self, dt, now):
        measured = self.io.measured_arm(max_age=.25)
        tracking = measured is not None and max(abs(a-b) for a,b in
            zip(self.io.arm_command()[:5], measured[:5])) < 12.
        if now - self.mirror.last_msg > self.mirror.m["timeout_s"] or not 0 < dt <= .15 or not tracking:
            self.stop_all()
            self.mirror.blocked = "stream/feedback lost; enable mirror again after recovery"
            self.log(self.mirror.blocked)
            return
        self.io.drive(0., 0., 0.)
        cmd = self.mirror.step(dt, now)
        if cmd:
            self.io.move_arm(cmd, duration_ms=110, wait=False, check=True)
