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
    c.last_status=clock[0];c.enable();clock[0]+=.401
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
        time.sleep(.6)
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
            time.sleep(.65)
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
