import hashlib
from explorer3.recording_readiness import evaluate,clock_check,ROOT


def evidence(robot='tank'):
    h={'software':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'agents'/robot).glob('*.py')},
       'monotonic_ns':10_000_000_000,'battery_voltage':7.6,'battery_received_ns':9_900_000_000,'motor_link':{'pending':False}}
    return {'health':h,'sensors':{'streams':{s:{'age_s':.02,'rate_hz':20} for s in ('rgb','depth','scan','imu')}},
        'machine':{'throttled':0},'clock':{'ok':True},'rtt_p95_ms':20}


def test_readiness_is_measurement_not_motion_acceptance():
    e=evidence();r=evaluate('tank',e,True,'Разрешён')
    assert r['ready'] and not any('physical_pass' in c for c in r['checks'])
    e['health']['motor_link']['pending']=True
    e['health']['fault']='motor_link_lost'
    assert not evaluate('tank',e,True,'Разрешён')['ready']


def test_all_red_checks_have_one_action_and_stale_camera_not_green():
    e=evidence();e['sensors']['streams']['rgb']['age_s']=2
    r=evaluate('tank',e,True,'Разрешён')
    camera=next(c for c in r['checks'] if c['key']=='rgb')
    assert camera['status']=='red' and camera['action']
    r=evaluate('rosmaster',{},False,'Выключен')
    assert not r['ready']
    assert all(c['action'] and '\n' not in c['action'] for c in r['checks'] if c['status']=='red')


def test_clock_includes_rtt_uncertainty_and_real_wall_offset():
    assert clock_check(1_000_000_000,1_010_000_000,1_005_000_000)['ok']
    assert not clock_check(1_000_000_000,1_050_000_000,1_025_000_000)['ok']
    assert not clock_check(1_000_000_000,1_002_000_000,970_000_000)['ok']
    assert not clock_check(1,2,None)['ok']


def test_agent_version_mismatch_and_low_rate_are_red():
    e=evidence();e['health']['software']['agent.py']='wrong';e['sensors']['streams']['rgb']['rate_hz']=7
    r=evaluate('tank',e,True,'Разрешён')
    assert {c['key'] for c in r['checks'] if c['status']=='red'}=={'version','rgb'}
