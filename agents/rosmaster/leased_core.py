"""Lease/STOP wrapper around the copied native BridgeCore; calibration is unchanged."""
import math
import struct
from rosmaster.core import BridgeCore
from rosmaster import link as L

NOT_READY=L.RESULT_NAMES.index('NOT_READY')


class LeasedCore(BridgeCore):
    # Session ends after this long without operator commands. Motion itself stops sooner: manual
    # input older than 0.4 s is treated as zero, and base commands carry a 240 ms controller lease.
    OPERATOR_TIMEOUT_S = 1.0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permitted = False
        self.operator_until = 0.
        self.stop_reason = 'Explicit enable required'
        self.joint_times = [None]*6
        self.joint_valid = [False]*6
        self.motion_commands_sent = 0
        self.arm_not_ready = 0
        self.clock_resyncs = 0
        self.clock_last_step_us = 0
        self.clock_max_step_us = 0
        self.clock_outlier = None

    # The copied BridgeCore measures the MCU clock offset only while connecting. Command leases are
    # 240 ms, so ordinary crystal drift (~100 ppm) makes every command EXPIRED after ~40 min in
    # "ready" (the original bridge restarts itself for this). Re-measure it periodically instead.
    RESYNC_S = 2.0

    def tick(self):
        now = self.clock()
        if self.phase == 'ready' and now-self.last_hello > self.RESYNC_S:
            self._hello(now)
        super().tick()

    def _send(self, kind, data=b'', **kwargs):
        result=super()._send(kind,data,**kwargs)
        if kind in (L.ARM,L.ARM_ENABLE) or (kind==L.BASE and any(struct.unpack_from('<fff',data))):
            self.motion_commands_sent+=1
        return result

    def ready(self):
        return (self.phase == 'ready' and self.last_status is not None
                and self.clock()-self.last_status < .25 and self.state.get('mode') == 1
                and not self.state.get('fault') and not self.state.get('arm_cancel_pending'))

    def enable(self):
        if not self.ready():raise RuntimeError('Controller is not ready or cancellation is pending')
        self.cmd = (0.,0.,0.)
        self.arm_target = None
        self.arm_dirty = False
        self.permitted = True
        self.operator_until = self.clock()+self.OPERATOR_TIMEOUT_S
        self.stop_reason = None

    def heartbeat(self):
        if self.clock()>=self.operator_until:self.disarm('Operator heartbeat expired')
        if not self.permitted:raise PermissionError('Session stopped; explicit enable required')
        self.operator_until = self.clock()+self.OPERATOR_TIMEOUT_S

    def cancel_arm(self):
        self.arm_target = None
        self.arm_dirty = False
        if self.link.boot_id and self.phase == 'ready':self._send(L.ARM_CANCEL)

    def disarm(self, reason):
        was_permitted = self.permitted
        self.permitted = False
        self.operator_until = 0.
        self.cmd = (0.,0.,0.)
        self.cmd_time = -1e9
        self.arm_target = None
        self.arm_dirty = False
        self.stop_reason = reason
        if was_permitted and self.link.boot_id:
            self._send(L.HOLD)
            self._send(L.ARM_CANCEL)

    def _set_phase(self, phase):
        if phase != 'ready':self.disarm('Controller state changed: '+phase)
        super()._set_phase(phase)

    def on_cmd_vel(self, vx, vy, wz):
        values=(vx,vy,wz)
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite base command')
        if any(values) and not self.permitted:raise PermissionError('Base is disarmed')
        super().on_cmd_vel(*values)

    def on_arm(self, servo_deg, runtime_ms):
        if not self.permitted:raise PermissionError('Arm is disarmed')
        if len(servo_deg)!=6 or not all(math.isfinite(v) for v in servo_deg):raise ValueError('Six finite joints required')
        super().on_arm(servo_deg,runtime_ms)

    def _stream(self, now):
        if self.permitted and now >= self.operator_until:self.disarm('Operator heartbeat expired')
        if not self.permitted:
            self.cmd=(0.,0.,0.);self.arm_target=None
        # Do not turn an unresolved cancellation into repeated enable attempts.
        if self.state.get('arm_cancel_pending'):
            saved=self.arm_target;self.arm_target=None
            try:super()._stream(now)
            finally:self.arm_target=saved
        else:super()._stream(now)

    def _handle(self, kind, payload, now):
        if kind==L.IDENTITY:
            before=self.link.mcu_offset_us
            super()._handle(kind,payload,now)
            step=self.link.mcu_offset_us-before
            if before and step:
                # One HELLO round trip can be delayed by USB scheduling (seen: 11 ms). Smooth small
                # corrections, ignore a single outlier, accept a large step once it is confirmed.
                if abs(step)>5000:
                    confirmed=self.clock_outlier is not None and abs(step-self.clock_outlier)<5000
                    self.clock_outlier=None if confirmed else step
                    if not confirmed:
                        self.link.mcu_offset_us=before;return
                else:
                    self.clock_outlier=None
                    self.link.mcu_offset_us=before+step//2;step//=2
                self.clock_resyncs+=1;self.clock_last_step_us=step
                self.clock_max_step_us=max(self.clock_max_step_us,abs(step))
            return
        super()._handle(kind,payload,now)
        if kind==L.SERVO:
            # Mark validity only after the native core stored the angle (it is read without this lock).
            _,joint,error,device,raw,_=struct.unpack_from('<QBBBHQ',payload)
            if 1<=joint<=6:
                self.joint_valid[joint-1]=error==0 and device==0 and 96<=raw<=4000 and self.servo_deg[joint-1] is not None
                self.joint_times[joint-1]=now
        if kind==L.RESULT:
            _,_,op,result,_,_=struct.unpack('<QQBBBB',payload)
            if result and op in (L.ARM,L.ARM_ENABLE,L.BASE) and self.permitted:
                # NOT_READY for the arm is the controller's normal answer while it finishes an arm
                # cancellation; the runtime bounds that state (time and count) and logs it.
                if result==NOT_READY and op in (L.ARM,L.ARM_ENABLE):
                    self.arm_not_ready+=1
                else:self.disarm('Controller rejected command: '+str(op)+' / '+str(result))

    def measured_arm(self, max_age=.25):
        now=self.clock()
        values=list(self.servo_deg)
        if all(valid and stamp is not None and 0<=now-stamp<max_age and value is not None
               for valid,stamp,value in zip(self.joint_valid,self.joint_times,values)):
            return values
        return None
