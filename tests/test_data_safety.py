import json,sqlite3
from types import SimpleNamespace
import pytest
from explorer3 import data_safety as safety,catalog,collection


@pytest.fixture
def storage(tmp_path,monkeypatch):
    from explorer3 import resources
    monkeypatch.setattr(resources,'DATA',tmp_path)
    monkeypatch.setattr(safety,'DATA',tmp_path)
    monkeypatch.setattr(catalog,'DATA',tmp_path)
    monkeypatch.setattr(collection,'DATA',tmp_path)
    catalog.initialize()
    folder=tmp_path/'datasets/example/episode';folder.mkdir(parents=True)
    (folder/'research.jsonl').write_text('{"fixture":true}\n')
    (folder/'video.mp4').write_bytes(b'fixture bytes only')
    catalog.add_episode(dict(id='episode',profile='example',dataset=str(folder),episode_index=0,
        session='s',scene='l',category='demonstration',outcome='success',frames=10))
    return tmp_path


def test_sqlite_backup_restore_and_incremental_external_hashes(storage,tmp_path_factory,monkeypatch):
    monkeypatch.setattr(safety,'independent_storage',lambda path:True)
    external=tmp_path_factory.mktemp('external')
    safety.configure_external(str(external))
    report=safety.backup_now()
    assert report['warning'] is None and report['copied_files']==2
    for relative,entry in report['files'].items():assert safety.sha256(external/relative)==entry['sha256']
    restored=storage/'restored.sqlite3'
    safety.restore_catalog(report['snapshot'],restored)
    with sqlite3.connect(restored) as db:assert db.execute('SELECT id FROM episodes').fetchall()==[('episode',)]
    with pytest.raises(ValueError,match='перезапись'):safety.restore_catalog(report['snapshot'],restored)
    again=safety.backup_now(force=True)
    assert again['copied_files']==0
    relative=next(iter(report['files']));(external/relative).write_bytes(b'changed externally')
    bad=safety.backup_now(force=True)
    assert 'изменилась' in bad['warning'] and (external/relative).read_bytes()==b'changed externally'
    assert safety.sha256(__import__('pathlib').Path(bad['snapshot'])/'catalog.sqlite3')==bad['catalog_sha256']


def test_no_external_disk_is_visible_warning(storage,monkeypatch):
    monkeypatch.setattr(safety,'external_destination',lambda:None)
    report=safety.backup_now()
    assert report['warning'] and report['external'] is None
    assert safety.status()['available']
    assert safety.backup_now()['snapshot']==report['snapshot']


def test_disk_start_and_finalize_reserve(monkeypatch):
    monkeypatch.setattr(safety.shutil,'disk_usage',lambda _:SimpleNamespace(free=7*1024**3))
    with pytest.raises(RuntimeError,match='8 ГБ'):safety.require_space()
    assert safety.require_space(starting=False)>0
    monkeypatch.setattr(safety.shutil,'disk_usage',lambda _:SimpleNamespace(free=1024**3))
    with pytest.raises(RuntimeError,match='завершается'):safety.require_space(starting=False)


def test_restart_aborts_attempt_without_resuming_or_deleting(storage):
    s=collection.create_series('so101_pick_place','so101_pick_place','op','fixture')
    collection.begin('unfinished',s['id']);collection.progress('unfinished',5,{'camera':5})
    result=collection.recover_after_restart()
    assert result=={'aborted':1,'readbacks_queued':0,'recordings_resumed':0}
    assert collection.summary(s['id'])['current']['aborted']==1
    assert (storage/'datasets/example/episode/video.mp4').exists()


def test_low_space_aborts_session_and_closes_source_before_finalize(storage,monkeypatch):
    from explorer3 import session as sessions,resources
    from explorer3.control import Control
    monkeypatch.setattr(resources,'DATA',storage)
    monkeypatch.setattr(safety.shutil,'disk_usage',lambda _:SimpleNamespace(free=1024**3))
    events=[]
    class Source:
        def start(self,**kw):return {}
        def begin_episode(self):pass
        def sample(self,*a,**kw):return {'teleoperation_phase':'following'}
        def close(self):events.append('closed');return []
    class Writer:
        id='disk-fixture'
        def __init__(self,*a,**kw):pass
        def add(self,sample):safety.require_space(starting=False)
        def finish(self,outcome,provenance):events.append(outcome)
    monkeypatch.setattr(sessions,'HardwareSource',lambda **kw:Source())
    monkeypatch.setattr(sessions,'EpisodeWriter',Writer)
    control=Control();control.stop_event.clear();control.expires=float('inf')
    session=sessions.Session(control);session.record_requested=True
    session._run('so101_pick_place','fixture','fixture')
    assert events==['closed','aborted'] and session.outcome=='aborted'
    assert control.stop_event.is_set() and 'резерв' in session.error
