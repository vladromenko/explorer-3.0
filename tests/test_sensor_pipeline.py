import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agents.tank.sensors import SensorStore,decode_image,make_app
from explorer3.sensors import decode_packet,environment_features,SensorArchive,CHANNELS


def packet():
    store=SensorStore()
    store.put('depth',np.full((12,16),1200,np.uint16),{'unit':'mm'})
    store.put('scan',np.ones(360,np.float32),{'angle_min':0.,'angle_increment':np.pi/180,'range_min':.05,'range_max':12.})
    store.put('imu',np.arange(6,dtype=np.float32),{'units':['rad/s']*3+['m/s2']*3})
    client=TestClient(make_app(store,'test-secret'))
    response=client.get('/snapshot.npz',headers={'X-Explorer-Token':'test-secret'})
    assert response.status_code==200
    return decode_packet(response.content,received_ns=1_000_000_000),client


def test_observer_auth_and_no_motion_routes():
    p,client=packet()
    assert client.get('/health').status_code==403
    assert client.post('/command',headers={'X-Explorer-Token':'test-secret'},json={'v':1}).status_code==404
    assert p['meta']['read_only'] and p['meta']['motion_commands_sent']==0
    assert p['arrays']['depth'].dtype==np.uint16


def test_metric_features_and_stale_masks():
    p,_=packet();features=environment_features(p,p['received_ns'])
    assert len(features)==len(CHANNELS)==178
    assert np.allclose(features[:36],1.)
    assert np.allclose(features[72:120],1.2)
    assert np.allclose(features[120:168],1.)
    assert np.array_equal(features[168:174],np.arange(6))
    assert features[174]==1.
    stale=environment_features(p,p['received_ns']+11_000_000_000)
    assert np.all(stale[:175]==0) and np.all(stale[175:]==10)


def test_invalid_ranges_depth_missing_and_imu_never_become_valid_zero():
    p,_=packet();p['arrays']['scan'][:]=np.nan;p['arrays']['depth'][:]=0
    p['arrays']['imu'][0]=np.inf
    features=environment_features(p,p['received_ns'])
    assert np.isfinite(features).all() and np.all(features[:175]==0)
    del p['meta']['streams']['depth'];del p['arrays']['depth']
    assert environment_features(p,p['received_ns'])[-2]==10


def test_archive_roundtrip_preserves_raw_and_excludes_training(tmp_path):
    p,_=packet();writer=SensorArchive(tmp_path/'record')
    assert writer.add(p) and not writer.add(p)
    writer.finish();manifest=json.loads((writer.root/'manifest.json').read_text())
    assert manifest['frames']==1 and not manifest['training_eligible']
    restored=decode_packet((writer.root/'000000.npz').read_bytes())
    assert np.array_equal(restored['arrays']['depth'],p['arrays']['depth'])
    assert restored['meta']==p['meta']
    with pytest.raises(RuntimeError):writer.add(p)


def test_depth_row_padding_endianness_and_rgb_order():
    msg=SimpleNamespace(width=2,height=2,step=6,encoding='mono16',is_bigendian=True,
        data=b'\x03\xe8\x07\xd0xx\x0b\xb8\x0f\xa0xx')
    assert decode_image(msg).tolist()==[[1000,2000],[3000,4000]]
    msg=SimpleNamespace(width=1,height=1,step=4,encoding='bgr8',is_bigendian=False,data=bytes([1,2,3,0]))
    assert decode_image(msg).tolist()==[[[3,2,1]]]


def test_readonly_source_has_no_publisher_serial_or_subprocess():
    import ast
    from pathlib import Path
    tree=ast.parse(Path('agents/tank/sensors.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden={'create_publisher','publish','Serial','Popen','system'}
    assert not {getattr(c.func,'attr',getattr(c.func,'id','')) for c in calls}&forbidden


def test_real_act_uses_environment_input_without_training():
    import torch
    from lerobot.configs.types import FeatureType,PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    torch.set_num_threads(2);torch.manual_seed(19)
    cfg=ACTConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(2,)),
        'observation.environment_state':PolicyFeature(FeatureType.ENV,(178,)),
        'observation.images.front':PolicyFeature(FeatureType.VISUAL,(3,64,64))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(2,))},
        pretrained_backbone_weights=None,dim_model=32,n_heads=4,dim_feedforward=64,
        n_encoder_layers=1,n_decoder_layers=1,chunk_size=2,n_action_steps=1,use_vae=False,device='cpu')
    policy=ACTPolicy(cfg).eval()
    batch={'observation.state':torch.zeros(1,2),'observation.environment_state':torch.zeros(1,178),
           'observation.images.front':torch.zeros(1,3,64,64)}
    with torch.inference_mode():
        a=policy.predict_action_chunk(batch)
        batch['observation.environment_state'][:,:36]=2.
        b=policy.predict_action_chunk(batch)
    assert a.shape==(1,2,2) and torch.isfinite(b).all() and not torch.allclose(a,b)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_multimodal_fixture_official_dataset_roundtrip(tmp_path,monkeypatch):
    from explorer3 import recording,catalog,profiles
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    monkeypatch.setattr(recording,'DATA',tmp_path)
    monkeypatch.setattr(catalog,'add_episode',lambda row:None)
    cfg={**profiles.PROFILES['tank_dock_rgbd_lidar'],'domain':'sim'}
    monkeypatch.setitem(profiles.SIM_PROFILES,'sim_sensor_fixture',cfg)
    writer=recording.EpisodeWriter('sim_sensor_fixture','fixture','fixture',domain='fixture')
    p,_=packet()
    for i in range(4):
        now=p['received_ns']+i*50_000_000
        writer.add({'observation_timestamp_ns':now,'observation.state':[0.,0.],'action':[0.,0.],
            'action_send_start_ns':now+1,'action_send_end_ns':now+2,
            'observation.environment_state':environment_features(p,now).tolist(),'_sensor_packet':p,
            'images':{'front':{'image':np.zeros((400,640,3),np.uint8),'timestamp_ns':now,'sequence':i}},
            'command_source':'human','action_acknowledged':False})
    writer.finish('aborted',{'fixture':True})
    ds=LeRobotDataset('local/fixture_readback',root=writer.root,video_backend='pyav')
    assert len(ds)==4 and ds[0]['observation.environment_state'].shape==(178,)
    assert np.allclose(ds[0]['observation.environment_state'].numpy(),environment_features(p,p['received_ns']))
    rows=[json.loads(line) for line in (writer.root/'research.jsonl').read_text().splitlines()]
    assert {r['sensor_packet'] for r in rows}=={'sensors/000000.npz'}
    assert all('_sensor_packet' not in r for r in rows)
    assert json.loads((writer.root/'provenance.json').read_text())['synthetic']
    from explorer3 import validation,datasets
    monkeypatch.setattr(datasets,'DATA',tmp_path)
    monkeypatch.setattr(validation,'DATA',tmp_path);(tmp_path/'validation').mkdir()
    monkeypatch.setattr(catalog,'rows',lambda table:[{'id':writer.id,'dataset':str(writer.root),'frames':4,'outcome':'aborted'}])
    assert validation.validate(writer.id,domain='fixture')['readback']=='PASS'


def test_multimodal_profile_cannot_bypass_incident(tmp_path,monkeypatch):
    from explorer3 import trials,local_work
    monkeypatch.setattr(trials,'DATA',tmp_path)
    monkeypatch.setattr(local_work,'require_hardware_mode',lambda robot:None)
    (tmp_path/'validation').mkdir();(tmp_path/'validation/tank-motion-incident.json').write_text('{}')
    with pytest.raises(PermissionError,match='STOP Tank'):
        trials.require_physical({'profile':'tank_dock_rgbd_lidar'})


def test_smolvla_cannot_silently_ignore_environment():
    from explorer3.policy_catalog import validate_recipe
    with pytest.raises(ValueError,match='environment_state'):validate_recipe('smolvla','tank_dock_rgbd_lidar')


def test_lidar_reuses_native_params_and_refuses_competing_owner(tmp_path,monkeypatch):
    from agents.tank import lidar
    driver=tmp_path/'driver';driver.touch();device=tmp_path/'device';device.touch()
    monkeypatch.setattr(lidar,'DRIVER',driver);monkeypatch.setattr(lidar,'DEVICE',device)
    monkeypatch.setattr(lidar.subprocess,'run',lambda *args,**kw:SimpleNamespace(returncode=0,stdout=b'123',stderr=b''))
    monkeypatch.setattr(lidar.subprocess,'Popen',lambda *args,**kw:pytest.fail('Must not open an owned lidar'))
    with pytest.raises(RuntimeError,match='owned'):lidar.Lidar().start(tmp_path)
    args=lidar.command()
    assert 'device_model:=MS200' in args and 'baudrate:=230400' in args and 'motor_speed:=15' in args
    assert 'scan_topic:=/explorer3/tank/scan_raw' in args
