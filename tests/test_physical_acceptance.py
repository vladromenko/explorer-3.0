import json,math,time
from explorer3.physical_acceptance import evaluate,Run,normalize
from explorer3 import physical_acceptance as physical


def samples():
    return [{'at':i*.1,'command':[.1,0] if i<20 else [0,0],'armed':i<20,'fault':None,
             'battery_v':7.8,'battery_fresh':True,'yaw_rate_rad_s':0.,
             'scan_fresh':True,'camera_fresh':True,'depth_fresh':True,'imu_fresh':True,
             'camera_sequence':i} for i in range(40)]


def test_stop_needs_operator_and_disarm_not_just_zero_command():
    rows=samples()
    assert evaluate('tank','stop',rows,False)['result']=='FAIL'
    result=evaluate('tank','stop',rows,True)
    assert result['result']=='PASS' and not result['motor_acknowledgement']
    for row in rows:row['armed']=True
    assert evaluate('tank','stop',rows,True)['result']=='FAIL'
    rows=samples();rows[-1]['fault']='motor_link_lost'
    assert evaluate('tank','stop',rows,True)['result']=='FAIL'


def test_charger_requires_confirmation_time_voltage_and_trend():
    rows=samples()
    assert evaluate('tank','charger',rows,True)['result']=='FAIL' # too short
    rows=[{**rows[0],'at':i*.1} for i in range(110)]
    assert evaluate('tank','charger',rows,True)['result']=='PASS'
    assert evaluate('tank','charger',rows,False)['result']=='FAIL'
    rows[-1]['battery_v']=8.4
    assert evaluate('tank','charger',rows,True)['result']=='FAIL'


def test_straight_imu_requires_calibration_when_yaw_exceeds_ten_degrees():
    rows=samples()
    for r in rows:r['yaw_rate_rad_s']=math.radians(4)
    result=evaluate('tank','straight',rows,True)
    assert result['result']=='FAIL'
    assert any('калибровка' in c['action'] for c in result['checks'])
    assert evaluate('tank','straight',samples(),True)['result']=='PASS'


def test_tank_lidar_step_does_not_accept_other_guard_reason():
    rows=samples();rows[10]['guard']='Наклон больше 20°'
    assert evaluate('tank','lidar',rows,True)['result']=='FAIL'
    rows[10]['guard']='Препятствие впереди'
    assert evaluate('tank','lidar',rows,True)['result']=='PASS'


def test_run_persists_failure_and_retry_without_permanent_permission_change(tmp_path,monkeypatch):
    monkeypatch.setattr(physical,'DATA',tmp_path)
    run=Run('so101',lambda:None)
    run.current='ownership';run.samples=[{'at':i,'owner':'legacy_8020'} for i in range(3)]
    assert run.finish(True)['result']=='FAIL'
    run.current='ownership';run.samples=[{'at':i,'owner':'explorer'} for i in range(3)]
    assert run.finish(True)['result']=='PASS'
    record=json.loads(run.path.read_text())
    assert len(record['attempts'])==2 and not record['complete']
    assert record['physical_commands_from_wizard']==0
    assert not (tmp_path/'configs/fleet.json').exists()


def test_native_imu_units_and_sensor_freshness():
    h={'monotonic_ns':1_000_000_000,'imu':[0,0,1,0,0,90],
       'imu_received_ns':990_000_000,'battery_received_ns':990_000_000,'battery_voltage':7.8}
    value=normalize('tank',h)
    assert math.isclose(value['yaw_rate_rad_s'],math.pi/2)
    assert not value['scan_fresh'] and value['battery_fresh']
