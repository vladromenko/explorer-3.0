import copy
import numpy as np
import pytest
from explorer3.fleet_live import ClockAlignment,frame_id,readiness,match_observations
from explorer3.sensors import environment_features,ROSMASTER_CHANNELS,SensorArchive
from test_sensor_pipeline import packet


def sample(before=1_000_000_000,after=1_020_000_000,boot='boot'):
    return {'before_ns':before,'after_ns':after,'remote':{'boot_id':boot,'clock_pair':{
        'monotonic_ns':(before+after)//2-500_000_000,'ros_time_ns':(before+after)//2+9_500_000_000,
        'uncertainty_ns':1000,'use_sim_time':False}}}


def test_clock_offset_uncertainty_stamp_expiry_and_boot():
    clock=ClockAlignment();s=sample();clock.add(s,wall_offset_ns=100_000_000_000)
    est=clock.estimate(s['remote'],1_020_000_000)
    assert est['offset_ns']==500_000_000 and est['uncertainty_ns']==10_001_000
    stamp,kind=clock.stamp(s['remote'],{'source_timestamp_ns':10_500_000_000},1_020_000_000)
    assert stamp==101_000_000_000 and kind=='source_header'
    with pytest.raises(ValueError,match='expired'):clock.estimate(s['remote'],20_000_000_000)
    newer=sample(2_000_000_000,2_002_000_000,'reboot');clock.add(newer,100_000_000_000)
    with pytest.raises(ValueError,match='boot'):clock.estimate(s['remote'],2_002_000_000)


def test_clock_jumps_and_sim_time_rejected():
    clock=ClockAlignment();s=sample();clock.add(s,100_000_000_000)
    bad=copy.deepcopy(s);bad['remote']['clock_pair']['ros_time_ns']+=1_000_000_000
    with pytest.raises(ValueError,match='jumped'):clock.estimate(bad['remote'],1_020_000_000)
    clock.add(bad,100_000_000_000);assert len(clock.samples)==1
    bad['remote']['clock_pair']['use_sim_time']=True
    with pytest.raises(ValueError,match='unavailable'):clock.add(bad)


def test_namespace_and_readiness_never_authorize_motors():
    assert frame_id('tank','/base_link')=='tank/base_link'
    assert frame_id('rosmaster','base_link')!='tank/base_link'
    with pytest.raises(ValueError):frame_id('tank','../cmd_vel')
    good={'uncertainty_ms':12,'ages_s':dict.fromkeys(['rgb','depth','scan','imu'],.1)}
    status={'running':True,'robots':{'tank':copy.deepcopy(good),'rosmaster':good}}
    result=readiness(status)
    assert result['observations_ready'] and not result['motion_authorized'] and not result['shared_map_ready']
    status['robots']['tank']['ages_s']['scan']=2
    assert not readiness(status)['observations_ready']
    status['running']=False
    assert 'Общая шина не отвечает' in readiness(status)['problems']


def test_rosmaster_features_preserve_two_lidars_and_missing_masks(tmp_path):
    p,_=packet();p['meta']['robot']='rosmaster'
    p['meta']['streams']['scan_rear']=p['meta']['streams']['scan'].copy()
    p['arrays']['scan_rear']=np.full(360,2,np.float32)
    f=environment_features(p,p['received_ns'])
    assert f.shape==(len(ROSMASTER_CHANNELS),)==(265,)
    assert np.allclose(f[:36],1) and np.allclose(f[178:214],2)
    assert np.all(f[-14:]==0)
    archive=SensorArchive(tmp_path/'rosmaster',robot='rosmaster')
    # No cross-platform data can be silently relabelled.
    tank,_=packet()
    with pytest.raises(ValueError,match='mismatch'):archive.add(tank)


def test_pairing_includes_uncertainty_and_rejects_stale_samples():
    a={'timestamp_ns':1_000_000_000,'uncertainty_ns':10_000_000,'age_s':.1,'packet_id':'a'}
    b={**a,'timestamp_ns':1_050_000_000,'packet_id':'b'}
    result=match_observations({'tank':[a],'rosmaster':[b]})
    assert result['matched'] and result['worst_case_skew_ms']==70 and not result['hardware_synchronized']
    b['uncertainty_ns']=50_000_000
    assert not match_observations({'tank':[a],'rosmaster':[b]})['matched']
    b['age_s']=2
    assert not match_observations({'tank':[a],'rosmaster':[b]})['matched']
    assert not match_observations({'rosmaster':[b]})['matched']


def test_status_file_ages_continue_to_grow_after_hub_stops(tmp_path,monkeypatch):
    import json
    from explorer3 import fleet_live
    monkeypatch.setattr(fleet_live,'DATA',tmp_path)
    monkeypatch.setattr(fleet_live.time,'time',lambda:110.)
    (tmp_path/'ros2-hub').mkdir()
    (tmp_path/'ros2-hub/status.json').write_text(json.dumps({'updated_at':100.,'robots':{
        'tank':{'ages_s':{'rgb':.1},'uncertainty_ms':5.}}}))
    value=fleet_live.read_hub_status()
    assert not value['running'] and value['robots']['tank']['ages_s']['rgb']==10.1
    assert value['robots']['tank']['uncertainty_ms']==7.5
