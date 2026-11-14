import math
import pytest
from agents.tank.limits import load,clip,Ramp,guard,sector_distance,tilt_degrees


def test_limits_apply_to_all_sources_even_without_incident(tmp_path):
    limits=load(tmp_path)
    assert clip(limits,1.,-3.,'manual')==(.2,-.5)
    for source in ('policy','replay','unknown'):
        assert clip(limits,1.,-3.,source)==(.3,-1.)
    limits['manual']={'v':10.,'w':10.};limits['absolute']={'v':100.,'w':100.}
    assert clip(limits,100.,100.,'manual')==(.2,.5)
    with pytest.raises(ValueError):clip(limits,math.nan,0.)


def test_acceleration_bounded_but_zero_immediate(tmp_path):
    clock=[0.];ramp=Ramp(load(tmp_path),clock=lambda:clock[0])
    for step in range(1,5):
        clock[0]=step*.05
        assert ramp.apply(.2,.5)==pytest.approx((min(step*.025,.2),min(step*.1,.5)))
    # No elapsed time needed for release, STOP or a safety stop.
    assert ramp.apply(0.,0.)==(0.,0.)
    clock[0]=100.
    assert ramp.apply(.2,.5)==pytest.approx((.05,.2))
    ramp.reset();assert ramp.previous==(0.,0.)


def test_local_directional_lidar_tilt_and_battery(tmp_path):
    value=load(tmp_path)
    sensors=dict(battery=7.5,battery_age=.1,tilt=0.,imu_age=.1,front=.1,rear=1.,scan_age=.1)
    assert guard(value,.2,0.,**sensors)[0]==(0.,0.)
    assert guard(value,-.2,0.,**sensors)[0]==(-.2,0.)
    assert guard(value,.1,0.,**sensors)[0]==(.1,0.)
    assert guard(value,.1,0.,**{**sensors,'front':.04})[0]==(0.,0.)
    for changed in ({'tilt':21.},{'battery':6.89},{'scan_age':1.},{'imu_age':1.},{'rear':None}):
        assert guard(value,-.2,0.,**{**sensors,**changed})[0]==(0.,0.)
    assert tilt_degrees([0,0,9.81])==0.
    assert tilt_degrees([9.81,0,0])==90.
    assert sector_distance([1.,.1,2.],-.2,.2,.05,10.)==.1
    assert sector_distance([1.,.1,2.],-.2,.2,.05,10.,rear=True) is None


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


def test_motor_link_recovery_only_writes_neutral_and_requires_ack(tmp_path,monkeypatch):
    import time
    from agents.tank import agent
    monkeypatch.setattr(agent,'ROOT',tmp_path)
    monkeypatch.setattr(agent,'controller_present',lambda:True)
    calls=[]
    class Driver:
        process=None
        def start(self):calls.append('open')
        def stop(self):calls.append('zero')
        def close(self):calls.append('close')
    monkeypatch.setattr(agent,'NativeDriver',Driver)
    h=agent.Hardware();h.done.set();h.watchdog.join(timeout=1)
    h.armed=True;h.lease='old';h.expires=time.monotonic()+1
    h.latch_fault('EIO')
    assert not h.armed and h.lease is None
    assert h.recover_link()
    assert calls==['open','zero','zero','zero','close']
    assert h.link.pending and h.fault and not h.armed
    with pytest.raises(PermissionError):h.command('old',2,.1,0.)
    with pytest.raises(PermissionError):h.acknowledge_link(False)
    assert h.acknowledge_link(True)['motor_link']['pending'] is False
    assert not h.armed and h.lease is None


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


def test_startup_with_saved_fault_is_inert(tmp_path,monkeypatch):
    import time
    from agents.tank import agent
    from agents.tank.motor_link import LinkState
    state=LinkState(tmp_path);state.lost('previous USB loss')
    monkeypatch.setattr(agent,'ROOT',tmp_path)
    monkeypatch.setattr(agent,'NativeDriver',lambda:pytest.fail('Startup must not open the motor driver'))
    monkeypatch.setattr(agent,'controller_present',lambda:True)
    h=agent.Hardware()
    try:
        time.sleep(.05)
        assert not h.armed and h.board is None and h.link.pending and not h.recovery_enabled
    finally:h.done.set();h.watchdog.join(timeout=1)


def test_docking_profile_reaches_agent_guard_without_relaxing_speed_condition(tmp_path,monkeypatch):
    from agents.tank import agent
    import time
    monkeypatch.setattr(agent,'ROOT',tmp_path)
    h=agent.Hardware();h.done.set();h.watchdog.join(1)
    h.task_profile='tank_dock'
    h.battery=7.8;h.imu=[0,0,1,0,0,0]
    h.battery_time=h.imu_time=h.scan_time=time.monotonic_ns()
    h.scan_front=h.scan_rear=.08
    assert h.guarded(.1,0)==((.1,0),None)
    assert h.guarded(.2,0)[0]==(0.,0.)
    h.task_profile=''
    assert h.guarded(.1,0)[0]==(0.,0.)
