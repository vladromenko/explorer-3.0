import math
import pytest
from agents.tank.limits import load,clip,Ramp,guard,sector_distance,tilt_degrees








def test_usb_state_is_recoverable_and_never_claims_physical_stop(tmp_path):
    from agents.tank.motor_link import LinkState
    state=LinkState(tmp_path);state.lost('EIO')
    assert state.status()['pending'] and not state.status()['physical_stop_confirmed']
    assert not (tmp_path/'incident-block.json').exists()
    with pytest.raises(RuntimeError):state.acknowledge(True)
    state.neutral_restored(3)
    with pytest.raises(PermissionError):state.acknowledge(False)
    assert LinkState(tmp_path).status()['recovered'] is False
    state.acknowledge(True)
    assert not state.status()['pending']
    for _ in range(4):state.lost('unplug');state.neutral_restored(3);state.acknowledge(True)
    assert state.status()['frequent_failures']




def test_udev_monitor_filters_controller_events():
    from io import StringIO
    from types import SimpleNamespace
    from agents.tank.motor_link import UdevMonitor
    events=[]
    monitor=UdevMonitor(lambda:events.append('removed'))
    monitor.process=SimpleNamespace(stdout=StringIO('ACTION=remove\nID_VENDOR_ID=1a86\nID_MODEL_ID=55d4\n\nACTION=add\nID_VENDOR_ID=1a86\nID_MODEL_ID=55d4\n\n'))
    monitor.read()
    assert events==['removed']


def test_front_scan_response_is_not_command_odometry():
    import time
    import numpy as np
    from explorer3.tank_response import ScanResponse
    now=time.monotonic_ns();angles=np.linspace(-.3,.3,61)
    def packet(distance,stamp):
        return {'received_ns':stamp,'request_elapsed_s':0.,'arrays':{'scan':distance/np.cos(angles),'imu':np.zeros(6)},
            'meta':{'boot_id':'test','streams':{'scan':{'received_ns':stamp,'age_s':0.,'angle_min':-.3,
                'angle_increment':.01,'range_min':.05,'range_max':12.},'imu':{'age_s':0.}}}}
    estimate=ScanResponse()
    assert estimate.observe(packet(1.,now))['forward_displacement_lidar'] is None
    measured=estimate.observe(packet(.98,now+100_000_000))
    assert measured['lidar_response_valid'] and measured['forward_displacement_lidar']==pytest.approx(.02)
    assert estimate.observe(packet(.98,now+100_000_000))['forward_displacement_lidar'] is None


def test_sensor_packets_preserve_received_imu_and_scan_history():
    import numpy as np
    from agents.tank.sensors import SensorStore
    store=SensorStore()
    store.put('imu',np.arange(6),{'source_timestamp_ns':100,'source_clock':'tank_ros_time'})
    store.put('imu',np.arange(6)+10,{'source_timestamp_ns':200,'source_clock':'tank_ros_time'})
    store.put('scan',np.ones(20),{'source_timestamp_ns':300})
    meta,arrays=store.capture()
    assert arrays['imu_history'].shape==(2,6)
    assert arrays['imu_history_source_ns'].tolist()==[100,200]
    assert arrays['imu_history_received_ns'][1]>arrays['imu_history_received_ns'][0]
    assert meta['streams']['imu']['frame_count']==2 and meta['streams']['imu']['rate_hz']>0
    assert 'imu_history' not in store.capture(include_history=False)[1]






def test_lost_motor_link_writes_neutral_as_soon_as_the_board_is_back(monkeypatch, tmp_path):
    """15.11.2026: a stall reset the board's USB (re-enumerated as ttyACM1); the agent kept a dead
    port and the board kept its last velocity. Now neutral x3 is written automatically when the board
    reappears, without any new motion lease."""
    import threading, time
    import agents.tank.agent as tank
    from agents.tank.motor_link import LinkState
    writes = []

    class Driver:
        def start(self): pass
        def stop(self): writes.append('neutral')
        def close(self): pass
    present = {'v': False}
    monkeypatch.setattr(tank, 'controller_present', lambda: present['v'])
    monkeypatch.setattr(tank, 'NativeDriver', Driver)
    hw = tank.Hardware.__new__(tank.Hardware)
    hw.lock, hw.lifecycle = threading.RLock(), threading.Lock()
    hw.link = LinkState(tmp_path)
    hw.board, hw.fault, hw.armed, hw.lease = None, None, True, 'x'
    hw.recovery_needed, hw.recovery_thread, hw.recovery_enabled = False, None, False
    hw.ramp = tank.Ramp(tank.load()) if hasattr(tank, 'load') else type('R', (), {'reset': lambda self: None})()
    hw.latch_fault('write failed: EIO')
    assert hw.armed is False and hw.lease is None
    time.sleep(.5)
    assert writes == []                              # board absent: nothing to write yet
    present['v'] = True
    for _ in range(50):
        if writes:
            break
        time.sleep(.1)
    assert writes == ['neutral'] * 3 and hw.recovery_needed is False and hw.armed is False


def test_autonomous_sessions_always_stop_before_obstacles():
    """Manual driving keeps the optional stop off; swarm/policy sessions stop 0.25 m from the lidar
    (0.17 m to the nose + margin) — the chair hit of 15.11.2026 happened with the stop off."""
    from agents.tank.limits import DEFAULTS, guard
    value = json_copy(DEFAULTS)
    assert guard(value, .2, 0., front=.22)[1] is None                       # manual: off by default
    cmd, why = guard(value, .2, .3, front=.22, autonomous=True)
    assert cmd == (0., .3) and why
    assert guard(value, .2, 0., front=.4, autonomous=True)[1] is None
    assert guard(value, -.1, 0., front=.1, rear=.5, autonomous=True)[1] is None   # backing away is allowed


def json_copy(v):
    import json
    return json.loads(json.dumps(v))
