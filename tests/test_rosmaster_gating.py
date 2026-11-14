"""Every entry point shares the operator fleet config, not historic stage files."""
import json
from explorer3 import local_work, fleet_config


def test_stale_stage_flags_cannot_grant_or_deny_robots(tmp_path,monkeypatch):
    monkeypatch.setattr(local_work,'DATA',tmp_path)
    cfg=tmp_path/'configs';cfg.mkdir()
    (cfg/'workflow.json').write_text(json.dumps({'stage':'A','allowed_robots':['tank']}))
    (cfg/'rosmaster-hardware-enabled.json').write_text('{"enabled":true}')
    assert not local_work.hardware_allowed('rosmaster')
    from explorer3.permission_migration import migrate
    migrate(tmp_path)
    assert local_work.hardware_allowed('rosmaster') and local_work.hardware_allowed('tank')
    assert not local_work.hardware_allowed('so101')
    fleet_config.update_robot('rosmaster',enabled=False,root=tmp_path)
    assert not local_work.hardware_allowed('rosmaster') and local_work.hardware_allowed('tank')
    assert local_work.workflow_stage() is None
    assert not local_work.hardware_allowed()


def test_all_disabled_means_homebrain_only(tmp_path,monkeypatch):
    from explorer3.permission_migration import migrate
    monkeypatch.setattr(local_work,'DATA',tmp_path)
    assert local_work.homebrain_only()
    migrate(tmp_path)
    assert not local_work.homebrain_only()
    for robot in fleet_config.ROBOTS:fleet_config.update_robot(robot,enabled=False,root=tmp_path)
    assert local_work.enabled()
