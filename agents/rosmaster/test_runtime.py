"""Run with the copied native modules on PYTHONPATH. No physical devices."""
import json
import os
import time
from pathlib import Path
import pytest
from rosmaster.fake_board import FakeBoard
from rosmaster import link as L
from leased_core import LeasedCore
from runtime import Runtime


def calibration():
    return json.loads(Path(os.environ['EXPLORER3_TEST_CALIBRATION']).read_text())


class Link:
    boot_id=42
    sequence=0
    def __init__(self):self.sent=[]
    def _request(self,kind,data=b'',**kwargs):
        self.sent.append((kind,data));self.sequence+=1;return self.sequence


def test_stop_expiry_and_reconnect_never_replay_arm_target():
    clock=[10.];link=Link()
    c=LeasedCore(link,calibration()['joints'],lambda *a:None,clock=lambda:clock[0])
    c.phase='ready';c.last_status=10.;c.state={'mode':1,'arm_enabled':True}
    c.enable();c.on_arm([90]*6,110);c._stream(clock[0])
    assert any(k==L.ARM for k,p in link.sent)
    c.disarm('STOP');before=len(link.sent)
    clock[0]+=.2;c._stream(clock[0])
    assert c.arm_target is None and not c.permitted
    assert all(k not in (L.ARM,L.ARM_ENABLE) for k,p in link.sent[before:])
    c._set_phase('searching');c._set_phase('ready')
    assert not c.permitted
    c.last_status=clock[0];c.enable();clock[0]+=1.001
    with pytest.raises(PermissionError):c.heartbeat()
    assert not c.permitted


def test_each_joint_requires_its_own_recent_valid_measurement():
    c=LeasedCore(Link(),calibration()['joints'],lambda *a:None,clock=lambda:10.)
    c.servo_deg=[90.]*6;c.joint_times=[9.9]*6;c.joint_valid=[True]*6
    assert c.measured_arm()==[90.]*6
    c.joint_times[5]=9.
    assert c.measured_arm() is None
    c.joint_times[5]=9.9;c.joint_valid[5]=False
    assert c.measured_arm() is None


def test_native_serial_watchdog_and_explicit_restart(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config/controller-calibration.json').write_text(json.dumps(calibration()))
    board=FakeBoard();runtime=Runtime(tmp_path,board.slave_path)
    try:
        runtime.start();end=time.monotonic()+5
        while time.monotonic()<end and not (runtime.core.ready() and runtime.voltage()):time.sleep(.03)
        assert runtime.core.ready() and runtime.core.motion_commands_sent==0
        session=runtime.enable('rosmaster_dock',observing=True)
        for i in range(6):
            runtime.command(session['lease'],i,{'drive':[.2,0,0]});time.sleep(.08)
        assert any(abs(vx)>0 for vx,vy,wz,t in board.base_commands)
        time.sleep(1.2)
        assert not runtime.armed and not any(board.velocity)
        assert not board.arm_targets
        with pytest.raises(PermissionError):runtime.command(session['lease'],9,{'drive':[1,0,0]})
        session=runtime.enable('rosmaster_dock',observing=True)
        runtime.command(session['lease'],0,{'drive':[.1,0,0]})
        runtime.stop('Operator STOP')
        assert not runtime.armed and runtime.core.arm_target is None
        assert any(kind==L.ARM_CANCEL for kind,payload in board.received)
    finally:runtime.close();board.close()


def test_transport_watchdog_does_not_wait_for_control_lock(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config/controller-calibration.json').write_text(json.dumps(calibration()))
    board=FakeBoard();runtime=Runtime(tmp_path,board.slave_path)
    try:
        runtime.start();end=time.monotonic()+5
        while time.monotonic()<end and not (runtime.core.ready() and runtime.voltage()):time.sleep(.03)
        session=runtime.enable('rosmaster_dock',observing=True)
        runtime.command(session['lease'],0,{'drive':[.2,0,0]})
        time.sleep(.12)
        with runtime.lock:
            time.sleep(1.25)
            assert not runtime.core.permitted
            assert not any(board.velocity)
    finally:runtime.close();board.close()


def ready_runtime(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config/controller-calibration.json').write_text(json.dumps(calibration()))
    board=FakeBoard();runtime=Runtime(tmp_path,board.slave_path)
    runtime.start();end=time.monotonic()+5
    while time.monotonic()<end and not (runtime.core.ready() and runtime.voltage()):time.sleep(.03)
    return board,runtime


def test_late_duplicate_command_is_dropped_and_first_stop_reason_kept(tmp_path):
    board,runtime=ready_runtime(tmp_path)
    try:
        session=runtime.enable('rosmaster_dock',observing=True)
        runtime.command(session['lease'],5,{'drive':[.1,0,0]})
        status=runtime.command(session['lease'],4,{'drive':[1,0,0]})
        assert status['armed'] and runtime.controls.input['drive']==[.1,0,0]
        runtime.stop('Battery telemetry stale (>30 s)');runtime.stop('Operator STOP')
        assert runtime.status()['reason']=='Battery telemetry stale (>30 s)'
        rows=[json.loads(l) for l in (tmp_path/'logs/control-sessions.jsonl').read_text().splitlines()]
        assert len(rows)==1 and rows[0]['reason'].startswith('Battery') and rows[0]['stale_commands']==1
    finally:runtime.close();board.close()


def test_armed_session_keeps_last_battery_value_while_reports_pause(tmp_path):
    board,runtime=ready_runtime(tmp_path)
    try:
        session=runtime.enable('rosmaster_dock',observing=True)
        runtime.battery_at-=5  # firmware stopped ADC reports while the base streams
        for i in range(4):
            runtime.command(session['lease'],i,{'drive':[.1,0,0]});time.sleep(.08)
        assert runtime.armed
        runtime.battery_at-=60
        runtime.command(session['lease'],9,{'drive':[0,0,0]});time.sleep(.12)
        assert not runtime.armed and 'Battery telemetry stale' in runtime.status()['reason']
    finally:runtime.close();board.close()


class StubCore:
    def __init__(self,clock):
        self.clock=clock;self.phase='ready';self.last_status=clock()
        self.state={'mode':1,'fault':0,'arm_enabled':True,'arm_cancel_pending':False}
        self.permitted=True;self.arm_target=None;self.arm_dirty=False;self.sent=[]
        self.joint_times=[clock()]*6;self.joint_valid=[True]*6;self.operator_until=clock()+1
        self.stop_reason=None;self.motion_commands_sent=0;self.measured=[90.,148.,12.,52.,90.,63.]
    def ready(self):return not self.state['arm_cancel_pending']
    def measured_arm(self,max_age=.25):return list(self.measured)
    def disarm(self,reason):self.permitted=False;self.stop_reason=reason;self.arm_target=None
    def on_arm(self,servo,ms):self.arm_target=(servo,ms);self.sent.append((list(servo),ms))


class StubControls:
    def __init__(self):
        import numpy as np
        class Jog:
            velocity=np.zeros(6);blocked=None
            def reset(self,servo):self.servo=list(servo)
        class Mirror:
            blocked=None
            def rebase(self):pass
        self.jog=Jog();self.mirror=Mirror();self.mode='manual';self.armed=True
        self._arm_tracking_blocked=False;self._arm_input_active=False;self.last_input=0
    def stop_all(self):self.armed=False


class StubIO:
    def __init__(self):self._arm_cmd=[90.]*6;self.base_stops=0
    def stop_base(self):self.base_stops+=1
    def control_step(self,dt):pass
    def measured_arm(self,max_age=.25):return None


def stub_runtime(tmp_path,profile='prepare'):
    clock=[100.]
    runtime=Runtime(tmp_path)
    runtime.core=StubCore(lambda:clock[0]);runtime.controls=StubControls();runtime.io=StubIO()
    runtime.lease='L';runtime.profile=profile
    runtime.session={'started':0,'started_monotonic':clock[0],'profile':profile,'arm_mode':'cartesian','commands':0,
                     'stale_commands':0,'max_command_gap':0.,'last_command':clock[0],'max_loop_dt':0.,'battery_at_enable':11.7}
    return runtime,clock


def test_prepare_streams_short_steps_and_never_runs_far_ahead(tmp_path):
    runtime,clock=stub_runtime(tmp_path)
    runtime.begin_pose([90]*6,runtime.core.measured,clock[0])
    for _ in range(10):
        clock[0]+=.05;runtime.prepare_step(clock[0],.05)
    commands=[c for c,ms in runtime.core.sent]
    assert all(runtime.POSE_STEP_MS<=ms<=600 for c,ms in runtime.core.sent)  # short steps, scaled by distance
    assert commands[0][1]==147. and abs(commands[-1][1]-148.)<=5.  # held near the 4 degree lead
    runtime.core.measured=[90.]*6
    clock[0]+=.05;runtime.prepare_step(clock[0],.05)
    assert runtime.lease is None and 'Поза достигнута' in runtime.reason


def test_controller_arm_cancel_recovers_from_measured_pose_a_bounded_number_of_times(tmp_path):
    runtime,clock=stub_runtime(tmp_path,'rosmaster_manual')
    runtime.core.arm_target=([100]*6,160)
    runtime.core.state['arm_cancel_pending']=True
    assert runtime.arm_recovering(clock[0]) and runtime.core.arm_target is None and runtime.lease=='L'
    clock[0]+=.3;runtime.core.state['arm_cancel_pending']=False
    assert not runtime.arm_recovering(clock[0])
    assert runtime.io._arm_cmd==runtime.core.measured and runtime.lease=='L'
    for i in range(runtime.ARM_CANCEL_MAX):
        clock[0]+=.1;runtime.core.state['arm_cancel_pending']=True;runtime.arm_recovering(clock[0])
        clock[0]+=.1;runtime.core.state['arm_cancel_pending']=False;runtime.arm_recovering(clock[0])
    assert runtime.lease is None and 'отменил движение руки' in runtime.reason
    rows=[json.loads(l) for l in (tmp_path/'logs/control-sessions.jsonl').read_text().splitlines()]
    assert sum(r['event']=='arm_cancel' for r in rows)==runtime.ARM_CANCEL_MAX+1


def test_spaced_arm_cancels_are_recovered_without_limit(tmp_path):
    runtime,clock=stub_runtime(tmp_path,'rosmaster_manual')
    for i in range(12):
        clock[0]+=3.;runtime.core.state['arm_cancel_pending']=True;runtime.arm_recovering(clock[0])
        clock[0]+=.2;runtime.core.state['arm_cancel_pending']=False;runtime.arm_recovering(clock[0])
    assert runtime.lease=='L' and runtime.session['recoveries']==12


def test_unresolved_arm_cancel_stops(tmp_path):
    runtime,clock=stub_runtime(tmp_path,'rosmaster_manual')
    runtime.core.state['arm_cancel_pending']=True
    runtime.arm_recovering(clock[0]);clock[0]+=runtime.ARM_RECOVERY_S+.1;runtime.arm_recovering(clock[0])
    assert runtime.lease is None and 'не снялась' in runtime.reason


def test_mcu_clock_offset_is_remeasured_while_ready(tmp_path):
    board,runtime=ready_runtime(tmp_path)
    try:
        assert runtime.core.phase=='ready'
        runtime.link.mcu_offset_us+=300_000  # simulated drift beyond the 240 ms lease
        end=time.monotonic()+2*runtime.core.RESYNC_S+2
        while time.monotonic()<end and runtime.core.clock_resyncs==0:time.sleep(.05)
        # a large step is accepted only after a second HELLO confirms it
        assert runtime.core.clock_resyncs>=1 and abs(runtime.core.clock_last_step_us+300_000)<20_000
        session=runtime.enable('rosmaster_dock',observing=True)
        runtime.command(session['lease'],0,{'drive':[.1,0,0]});time.sleep(.1)
        assert runtime.armed
    finally:runtime.close();board.close()


def test_single_clock_outlier_is_ignored_and_small_steps_smoothed():
    class Link:
        boot_id=7;mcu_offset_us=1_000_000;session=1;sequence=0
    import struct
    c=LeasedCore(Link(),calibration()['joints'],lambda *a:None,clock=lambda:1.)
    def identity(offset):
        def base(kind,payload,now):c.link.mcu_offset_us=offset
        return base
    from rosmaster.core import BridgeCore
    original=BridgeCore._handle
    try:
        BridgeCore._handle=lambda self,kind,payload,now:setattr(self.link,'mcu_offset_us',target[0])
        target=[1_011_000];c._handle(L.IDENTITY,b'',1.)
        assert c.link.mcu_offset_us==1_000_000
        target=[1_000_400];c._handle(L.IDENTITY,b'',1.)
        assert c.link.mcu_offset_us==1_000_200
        target=[1_300_000];c._handle(L.IDENTITY,b'',1.);assert c.link.mcu_offset_us==1_000_200
        target=[1_300_500];c._handle(L.IDENTITY,b'',1.);assert c.link.mcu_offset_us==1_300_500
    finally:BridgeCore._handle=original


class DirectIO(StubIO):
    def __init__(self):
        super().__init__();self.drives=[];self.moves=[];self.measured=[90.]*6
    def drive(self,vx,vy,wz,manual=False):self.drives.append((vx,vy,wz))
    def measured_arm(self,max_age=.25):return list(self.measured)
    def arm_command(self):return list(self._arm_cmd)
    def move_arm(self,servo,duration_ms=None,wait=True,check=True):self.moves.append(list(servo));self._arm_cmd=list(servo)


def test_policy_actions_are_shape_checked_rate_limited_and_never_leave_tracking(tmp_path):
    runtime,clock=stub_runtime(tmp_path,'rosmaster_pick_place');runtime.io=DirectIO();runtime.direct_source='policy'
    with pytest.raises(ValueError):runtime.validate_direct({'base':[1,0,0]})
    runtime.direct=(clock[0],runtime.validate_direct({'arm':[90,120,90,90,90,90]}))
    import time as t
    runtime.direct=(t.monotonic(),runtime.direct[1]);runtime.direct_step(.05,t.monotonic())
    assert runtime.io.moves[-1][1]==93.  # 60 deg/s * 0.05 s
    runtime.io.measured=[90.]*6;runtime.io._arm_cmd=[90,105,90,90,90,90]
    with pytest.raises(RuntimeError):runtime.direct_step(.05,t.monotonic())
    runtime.direct=(t.monotonic()-1,runtime.direct[1]);runtime.direct_step(.05,t.monotonic())
    assert runtime.io.drives[-1]==(0.,0.,0.)


def test_dock_policy_actions_are_capped_by_native_manual_limits(tmp_path):
    import time as t
    runtime,clock=stub_runtime(tmp_path,'rosmaster_dock');runtime.io=DirectIO();runtime.direct_source='policy'
    with pytest.raises(ValueError):runtime.validate_direct({'arm':[90]*6})
    runtime.direct=(t.monotonic(),runtime.validate_direct({'base':[5,-5,9]}));runtime.direct_step(.05,t.monotonic())
    b=runtime.cfg['manual_base'];assert runtime.io.drives[-1]==(b['max_vx'],-b['max_vy'],b['max_wz'])


def test_measured_arm_never_returns_missing_angles():
    c=LeasedCore(Link(),calibration()['joints'],lambda *a:None,clock=lambda:10.)
    c.servo_deg=[90.]*5+[None];c.joint_times=[9.9]*6;c.joint_valid=[True]*6
    assert c.measured_arm() is None


def test_near_limit_reading_is_clamped_far_outside_is_not(tmp_path):
    from runtime import NativeIO
    io=NativeIO.__new__(NativeIO)
    class R:pass
    io.runtime=R();io.runtime.core=type('C',(),{'measured_arm':lambda self,a=.25:[90.,129.5,-0.9,24.6,89.9,29.7]})()
    assert io.measured_arm()==[90.,130.,0.,25.,90.,30.]
    io.runtime.core=type('C',(),{'measured_arm':lambda self,a=.25:[90.,198.,-49.,24.,90.,36.]})()
    assert io.measured_arm()[1:3]==[198.,-49.]


def test_a_fault_in_one_step_ends_the_session_but_not_the_agent(tmp_path):
    board,runtime=ready_runtime(tmp_path)
    try:
        session=runtime.enable('rosmaster_dock',observing=True)
        original=runtime.controls.tick
        def broken(dt,now):
            runtime.controls.tick=original;raise ZeroDivisionError('test fault')
        runtime.controls.tick=broken
        runtime.command(session['lease'],0,{'drive':[0,0,0]});time.sleep(.4)
        assert not runtime.armed and 'Ошибка цикла' in runtime.status()['reason'] and runtime.error is None
        session=runtime.enable('rosmaster_dock',observing=True)
        for i in range(3):runtime.command(session['lease'],i,{'drive':[0,0,0]});time.sleep(.05)
        assert runtime.armed
    finally:runtime.close();board.close()


def test_folded_arm_starts_from_nearest_commandable_pose_with_long_enough_move(tmp_path):
    runtime,clock=stub_runtime(tmp_path)
    runtime.core.measured=[92.,135.,-49.,31.,90.,27.]
    runtime.begin_pose([90]*6,runtime.core.measured,clock[0])
    clock[0]+=.05;runtime.prepare_step(clock[0],.05)
    command,ms=runtime.core.sent[-1]
    assert command[2]==0. and command[5]==30.     # nearest commandable values, not -49 / 27
    assert ms>=2400                                # 49 degrees at 20 deg/s, not a 160 ms jump


def test_arm_targets_never_lead_the_measurement_by_more_than_the_cancel_margin():
    from runtime import NativeIO
    from cubebot import config
    from cubebot.arm_model import ArmModel
    cfg=config.load()
    class R:
        log=staticmethod(lambda m:None)
        def __init__(self):
            self.core=type('C',(),{'measured_arm':lambda s,a=.25:[90.,100.,90.,90.,90.,60.]})()
            self.sent=[]
    rt=R()
    io=NativeIO.__new__(NativeIO);io.runtime=rt
    from cubebot.robot_io import BaseIO
    BaseIO.__init__(io,cfg,ArmModel(cfg['arm']['tcp_offset_m']),lambda m:None)
    io._arm_cmd=[90.,100.,90.,90.,90.,60.]
    io._publish_arm=lambda servo,ms:rt.sent.append(list(servo))
    io.move_arm([90,80,90,90,90,90],duration_ms=160,wait=False,check=False)
    assert rt.sent[-1]==[90,96,90,90,90,64]


def test_arm_mode_switches_inside_the_session_without_stopping(tmp_path):
    board,runtime=ready_runtime(tmp_path)
    try:
        session=runtime.enable('rosmaster_dock',observing=True)
        runtime.command(session['lease'],0,{'drive':[0,0,0]},arm_mode='joint');time.sleep(.1)
        assert runtime.armed and runtime.controls.arm_mode=='joint'
        runtime.command(session['lease'],1,{'drive':[0,0,0]},arm_mode='cartesian');time.sleep(.1)
        assert runtime.armed and runtime.controls.arm_mode=='cartesian'
    finally:runtime.close();board.close()
