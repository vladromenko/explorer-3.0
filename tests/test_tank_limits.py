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




