import json
from types import SimpleNamespace
import pytest
from explorer3 import fleet_capture as f
from explorer3 import fleet_live


def test_capture_rejects_bad_duration_stale_hub_and_path(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'DATA',tmp_path)
    monkeypatch.setattr(fleet_live,'read_hub_status',lambda:{'running':False})
    for seconds in (True,0,120,'30'):
        with pytest.raises(ValueError):f.start(seconds)
    with pytest.raises(ValueError,match='шина'):f.start(15)
    for name in ('../a','A'*32,'a'*31):
        with pytest.raises(ValueError):f.entry(name)


def test_capture_container_is_observation_only(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'DATA',tmp_path)
    key='a'*32;(f.directory()/key).mkdir()
    cmd=f.command(key,15);text=' '.join(cmd)
    assert 'fastdds.xml' in text and 'fastdds-hub.xml' not in text
    assert '/dev' not in text and 'configs' not in text
    assert '--read-only' in cmd and '--cap-drop' in cmd
    assert 'capture_ros2_observations.py' in text


def test_capture_failure_finalizes_report_and_releases_slot(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'DATA',tmp_path);monkeypatch.setattr(f,'_active','a'*32)
    key='a'*32;p=f.directory()/key;p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'id':key,'phase':'recording','created_at':1,'training_eligible':False}))
    monkeypatch.setattr(f.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    f._run(key,15)
    assert f._active is None
    value=json.loads((p/'manifest.json').read_text())
    assert value['phase']=='failed' and not value['training_eligible']


def test_replay_drops_stale_views_and_retains_source_time(tmp_path):
    import numpy as np
    from explorer3.fleet_replay import ReplayIndex
    r=ReplayIndex(tmp_path);r.start=1_000_000_000
    r.latest['tank','rgb']=(np.zeros((8,8,3),np.uint8),{'received_ns':1_000_000_000,'source_timestamp_ns':900_000_000})
    r.render(1_100_000_000)
    view=r.frames[0]['views']['tank/rgb']
    assert view['source_timestamp_ns']==900_000_000 and view['age_s']==.2
    assert (tmp_path/view['file']).read_bytes()[:2]==b'\xff\xd8'
    r.render(3_000_000_000)
    assert not r.frames[1]['views']
    r.finish();result=json.loads((tmp_path/'replay.json').read_text())
    assert result['display_only'] and not result['hardware_synchronized']


def test_capture_status_survives_ui_restart_without_claiming_stale_success(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'DATA',tmp_path);monkeypatch.setattr(f,'_active',None)
    monkeypatch.setattr(f.time,'time',lambda:110.)
    key='a'*32;p=f.directory()/key;p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'id':key,'created_at':100.,'requested_seconds':30,'phase':'recording'}))
    assert f.state()['active']==key
    monkeypatch.setattr(f.time,'time',lambda:400.)
    assert f.state()['active'] is None and f.state()['records'][0]['phase']=='interrupted'
    assert json.loads((p/'manifest.json').read_text())['phase']=='recording'  # Original evidence retained.


def test_verified_file_is_not_lost_when_container_cleanup_times_out(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'DATA',tmp_path);monkeypatch.setattr(f,'_active','a'*32)
    key='a'*32;p=f.directory()/key;p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'id':key,'phase':'complete','readback':{'passed':True},'created_at':1}))
    def timeout(cmd,**kw):raise f.subprocess.TimeoutExpired(cmd,120)
    monkeypatch.setattr(f.subprocess,'run',timeout)
    f._run(key,30)
    value=json.loads((p/'manifest.json').read_text())
    assert value['phase']=='complete' and value['cleanup_warning'] and f._active is None
