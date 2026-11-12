"""Explorer adapter over the selected CubeBot controls; no ROS or device construction.

_manual_step and _mirror_step are verbatim copies from the rollback snapshot of
Console 8090. The wrapper only validates input, separates profiles and expires a
session. Native CartesianJog/MirrorMapper and their calibration are unchanged.
"""
import math
import numpy as np


class NativeControls:
    def __init__(self, io, cfg, arm_model, mirror_cfg, log=lambda message: None):
        from cubebot.arm_control import CartesianJog
        from cubebot.mirror import MirrorMapper
        self.io, self.cfg, self.arm, self.log = io, cfg, arm_model, log
        self.jog = CartesianJog(arm_model, cfg, io.arm_command())
        self.mirror = MirrorMapper(arm_model, cfg['arm'], mirror_cfg, io.arm_command())
        self.input = dict(drive=[0,0,0], arm=[0,0,0], pitch=0, roll=0, grip=0, slow=False)
        self.last_input = -float('inf')
        self._arm_input_active = False
        self._arm_tracking_blocked = False
        self.mode = 'manual'
        self.arm_mode = 'cartesian'
        self.profile = None
        self.armed = False

    def enable(self, profile, now):
        if self.armed:raise ValueError('Stop the previous control session first')
        if profile not in {'rosmaster_dock','rosmaster_pick_place','rosmaster_manual'}:raise ValueError('Unknown profile')
        if profile in {'rosmaster_pick_place','rosmaster_manual'}:
            measured=self.io.measured_arm(max_age=.25)
            if measured is None:raise ValueError('Fresh native arm feedback required; estimates are not measurements')
            self.io._arm_cmd=list(measured)
            self.jog.reset(measured)
        self.profile=profile;self.armed=True;self.last_input=now

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

    def leader(self, positions, now):
        return self.mirror.update(positions,now)

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

    def tick(self, dt, now):
        if self.armed:
            if not math.isfinite(dt) or not 0<dt<=.15 or now-self.last_input>=.4:
                self.stop_all()
            elif self.mode=='mirror':self._mirror_step(dt,now)
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
