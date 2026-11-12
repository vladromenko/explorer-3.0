"""Lease/STOP wrapper around the copied native BridgeCore; calibration is unchanged."""
import math
import struct
from rosmaster.core import BridgeCore
from rosmaster import link as L


class LeasedCore(BridgeCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permitted = False
        self.operator_until = 0.
        self.stop_reason = 'Explicit enable required'
        self.joint_times = [None]*6
        self.joint_valid = [False]*6
        self.motion_commands_sent = 0

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
        self.operator_until = self.clock()+.4
        self.stop_reason = None

    def heartbeat(self):
        if self.clock()>=self.operator_until:self.disarm('Operator heartbeat expired')
        if not self.permitted:raise PermissionError('Session stopped; explicit enable required')
        self.operator_until = self.clock()+.4

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
        if kind==L.SERVO:
            _,joint,error,device,raw,_=struct.unpack_from('<QBBBHQ',payload)
            if 1<=joint<=6:
                self.joint_times[joint-1]=now
                self.joint_valid[joint-1]=error==0 and device==0 and 96<=raw<=4000
        super()._handle(kind,payload,now)
        if kind==L.RESULT:
            _,_,op,result,_,_=struct.unpack('<QQBBBB',payload)
            if result and op in (L.ARM,L.ARM_ENABLE,L.BASE) and self.permitted:
                self.disarm('Controller rejected command: '+str(op)+' / '+str(result))

    def measured_arm(self, max_age=.25):
        now=self.clock()
        if all(valid and stamp is not None and 0<=now-stamp<max_age
               for valid,stamp in zip(self.joint_valid,self.joint_times)):
            return list(self.servo_deg)
        return None
