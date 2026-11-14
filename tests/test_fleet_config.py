import json
import fcntl
from pathlib import Path
import pytest
from explorer3 import fleet_config as fleet, permission_migration as migration


def install(root,owner='explorer'):
    value={'version':1,'robots':{'so101':{'enabled':True,'owner':owner},'rosmaster':{'enabled':True},
        'tank':{'enabled':True,'mode':'floor'}},'training_authorized':False,'updated_by':'user','updated_at':0}
    fleet.atomic_write(fleet.path(root),value)
    return value


def test_only_fleet_controls_all_robots(tmp_path):
    for relative in migration.LEGACY:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    assert not fleet.robot_allowed('tank',tmp_path)[0]
    install(tmp_path)
    for robot in fleet.ROBOTS:
        assert fleet.robot_allowed(robot,tmp_path)[0]
        fleet.update_robot(robot,enabled=False,root=tmp_path)
        allowed,reason=fleet.robot_allowed(robot,tmp_path)
        assert not allowed and 'выключен' in reason
        fleet.update_robot(robot,enabled=True,root=tmp_path)
    fleet.update_robot('so101',owner='legacy_8020',root=tmp_path)
    assert not fleet.robot_allowed('so101',tmp_path)[0]
    assert fleet.robot_allowed('rosmaster',tmp_path)[0]
    fleet.path(tmp_path).write_text('{broken')
    assert not fleet.robot_allowed('tank',tmp_path)[0]


def test_migration_roundtrip_idempotent_and_no_legacy_service_calls(tmp_path):
    originals={name:json.dumps({'old_file':name}).encode() for name in migration.LEGACY}
    for name,content in originals.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    result=migration.migrate(tmp_path)
    assert result['status']=='completed'
    assert migration.migrate(tmp_path)==result
    assert all(not (tmp_path/name).exists() for name in originals)
    assert fleet.load(tmp_path)['robots']['so101']['owner']=='legacy_8020'
    assert fleet.robot_allowed('tank',tmp_path)[0] and fleet.robot_allowed('rosmaster',tmp_path)[0]
    assert not fleet.load(tmp_path)['training_authorized']
    archive=Path(result['archive'])
    assert all((archive/name).read_bytes()==content for name,content in originals.items())
    migration.rollback(tmp_path,archive)
    assert all((tmp_path/name).read_bytes()==content for name,content in originals.items())
    assert not fleet.path(tmp_path).exists()
    assert migration.rollback(tmp_path,archive)['status']=='rolled_back'


def test_rollback_preserves_new_user_decisions(tmp_path):
    result=migration.migrate(tmp_path)
    fleet.update_robot('tank',enabled=False,root=tmp_path)
    with pytest.raises(ValueError,match='изменён'):migration.rollback(tmp_path,result['archive'])
    assert not fleet.robot_allowed('tank',tmp_path)[0]


def test_migration_refuses_active_control(tmp_path):
    folder=tmp_path/'locks';folder.mkdir()
    with (folder/'physical-activity.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH)
        with pytest.raises(RuntimeError,match='завершите'):migration.migrate(tmp_path)
    assert not fleet.path(tmp_path).exists()


def test_interrupted_migration_can_restore_before_removing_legacy(tmp_path,monkeypatch):
    original=tmp_path/'configs/workflow.json';original.parent.mkdir();original.write_text('{"stage":"A"}')
    real_unlink=Path.unlink
    def interrupt(path,*args,**kwargs):
        if path==original:raise OSError('interrupted')
        return real_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupt)
    with pytest.raises(OSError):migration.migrate(tmp_path)
    monkeypatch.setattr(Path,'unlink',real_unlink)
    archive=next((tmp_path/'archive').iterdir())
    migration.rollback(tmp_path,archive)
    assert original.read_text()=='{"stage":"A"}' and not fleet.path(tmp_path).exists()


def test_disabling_robot_never_blocks_emergency_stop(tmp_path,monkeypatch):
    import httpx
    from explorer3 import rosmaster_control as rc, local_work
    install(tmp_path)
    fleet.update_robot('rosmaster',enabled=False,root=tmp_path)
    monkeypatch.setattr(local_work,'DATA',tmp_path)
    calls=[]
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200,json={'stopped':True})
    monkeypatch.setattr(rc,'_connection',lambda timeout:httpx.Client(base_url='http://fake',transport=httpx.MockTransport(handler)))
    assert rc.request('/stop',{})['stopped']
    with pytest.raises(PermissionError):rc.request('/command',{})
    assert calls==['/stop']
