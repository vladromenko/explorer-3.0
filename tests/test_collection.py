import json
import pytest
from explorer3 import collection,catalog


@pytest.fixture
def storage(tmp_path,monkeypatch):
    monkeypatch.setattr(catalog,'DATA',tmp_path)
    monkeypatch.setattr(collection,'DATA',tmp_path)
    catalog.initialize();collection.initialize()
    return tmp_path


def test_series_numbering_reused_layout_and_no_variant_mix(storage):
    a=collection.create_series('tank_dock','tank_dock_rgbd_lidar','operator','Старт слева')
    b=collection.create_series('tank_dock','tank_dock_rgbd_lidar','operator','Повтор',a['layout'])
    assert a['id']=='tank_dock-s01' and b['id']=='tank_dock-s02'
    assert a['layout']==b['layout']
    with pytest.raises(ValueError,match='другой вариант'):
        collection.create_series('tank_dock','tank_dock','operator','Новая')
    with pytest.raises(ValueError,match='не совпадают'):
        collection.context('tank_dock_rgbd_lidar',b['id'],'made-up-layout')
    assert collection.summary(b['id'])['independent_groups']==1


def test_zero_frame_abort_is_counted_without_fabricated_episode(storage):
    a=collection.create_series('so101_pick_place','so101_pick_place','operator','Кубик справа')
    collection.begin('attempt',a['id'])
    collection.completed('attempt','success',0)
    r=collection.summary(a['id'])
    assert r['current']=={'success':0,'failure':0,'aborted':1}
    assert r['to_50']==50 and catalog.rows('episodes')==[]


def test_nonmanaged_recording_context_is_optional(storage):
    assert collection.context('so101_pick_place','old-session','old-scene') is None
    with pytest.raises(ValueError):collection.create_series('tank_dock','rosmaster_dock','op','x')


def test_managed_episode_uses_existing_writer_and_automatic_readback(tmp_path):
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'tests/rosmaster_fake_agent.py','rosmaster_pick_place','managed'],
        cwd=root,env={**os.environ,'EXPLORER3_DATA':str(tmp_path),'PYTHONPATH':str(root)},
        capture_output=True,text=True,timeout=150)
    assert result.returncode==0,result.stderr[-3000:]+result.stdout[-2000:]
    report=json.loads(result.stdout.strip().splitlines()[-1])['collection_report']
    assert report['current']['success']==1 and report['independent_groups']==1
    attempt=report['attempts'][0];quality=json.loads(attempt['quality'])
    assert quality['status']=='pass',quality
    assert set(quality['thumbnails'])=={'first','last'}
    provenance=json.loads(next(tmp_path.glob('datasets/fixture/*/*/provenance.json')).read_text())
    assert provenance['hardware']['collection']['operator']=='fixture-operator'
    assert provenance['hardware']['collection']['battery_min_v']==11.7


@pytest.mark.parametrize('profile',['so101_pick_place','tank_dock_rgbd_lidar','rosmaster_dock_rgbd_lidar'])
def test_remaining_default_profiles_read_back_with_all_streams(tmp_path,profile):
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'tests/collection_writer_fixture.py',profile],cwd=root,
        env={**os.environ,'EXPLORER3_DATA':str(tmp_path),'PYTHONPATH':str(root)},capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr[-2000:]+result.stdout[-3000:]
    assert json.loads(result.stdout.strip().splitlines()[-1])['quality']=='pass'


def test_gap_is_quarantined_without_deleting_raw_recording(tmp_path):
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'tests/collection_writer_fixture.py','tank_dock_rgbd_lidar','gap'],cwd=root,
        env={**os.environ,'EXPLORER3_DATA':str(tmp_path),'PYTHONPATH':str(root)},capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr[-2000:]+result.stdout[-3000:]
    report=json.loads(next((tmp_path/'validation/collection').glob('*/quality.json')).read_text())
    assert report['status']=='warning' and any('200' in p for p in report['problems'])
