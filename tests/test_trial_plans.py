import json
from pathlib import Path
import pytest
from explorer3 import trials
from explorer3.recording import EpisodeWriter


def test_replay_never_moves_to_a_distant_start_pose(tmp_path,monkeypatch):
    row={'action':[90]*6}
    (tmp_path/'research.jsonl').write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(trials,'fingerprint',lambda path:'saved')
    replay=trials.ReplayActions({'dataset':str(tmp_path),'dataset_sha256':'saved'})
    with pytest.raises(ValueError,match='Начальная поза'):
        replay.next({'observation.state':[0]*6})
    assert replay.index==0
    assert replay.next({'observation.state':[90]*6})==[90]*6
    with pytest.raises(StopIteration):replay.next({'observation.state':[90]*6})


def test_missing_fleet_blocks_trial_before_loading_hardware(tmp_path,monkeypatch):
    monkeypatch.setattr(trials.local_work,'DATA',tmp_path)
    with pytest.raises(PermissionError,match='Допуск парка'):
        trials.require_physical({'profile':'so101_pick_place','id':'0'*32})


def test_recording_rejects_source_switch_before_writing_frame():
    writer=object.__new__(EpisodeWriter)
    writer.disk_checked_at=float('inf')  # This test isolates source changes, not storage admission.
    writer.camera='wrist';writer.category='rollout';writer.command_source='policy';writer.domain='real'
    sample={'images':{'wrist':{}},'command_source':'human'}
    with pytest.raises(ValueError,match='segment'):
        writer.add(sample)


def test_policy_bundle_includes_normalization_and_config(tmp_path):
    (tmp_path/'config.json').write_text('{}');(tmp_path/'model.safetensors').write_bytes(b'fixture')
    (tmp_path/'normalizer.json').write_text('{"mean":0}')
    plan={'checkpoint':str(tmp_path),'checkpoint_bundle_sha256':trials.checkpoint_hash(tmp_path)}
    trials.verify_policy_artifacts(plan)
    (tmp_path/'normalizer.json').write_text('{"mean":100}')
    with pytest.raises(ValueError,match='нормировка'):trials.verify_policy_artifacts(plan)
    with pytest.raises(ValueError,match='Старый план'):trials.verify_policy_artifacts({'checkpoint':str(tmp_path)})


def test_changed_trial_plan_invalidates_explicit_permission(tmp_path,monkeypatch):
    monkeypatch.setattr(trials,'DATA',tmp_path);monkeypatch.setattr(trials.local_work,'require_hardware_mode',lambda *_:None)
    (tmp_path/'trials').mkdir();(tmp_path/'configs').mkdir()
    plan={'id':'a'*32,'kind':'replay','profile':'rosmaster_dock','max_duration_s':1.}
    (tmp_path/'trials'/(plan['id']+'.json')).write_text(json.dumps(plan))
    trials.authorize(plan['id']);trials.require_physical(plan)
    with pytest.raises(PermissionError,match='План изменился'):trials.require_physical({**plan,'max_duration_s':60.})


def test_replay_refuses_incomplete_initial_measurement(tmp_path,monkeypatch):
    (tmp_path/'research.jsonl').write_text(json.dumps({'action':[90]*6})+'\n')
    monkeypatch.setattr(trials,'fingerprint',lambda _:'saved')
    replay=trials.ReplayActions({'dataset':str(tmp_path),'dataset_sha256':'saved'})
    for state in ([90]*5,[90]*5+[float('nan')]):
        with pytest.raises(ValueError,match='шести'):replay.next({'observation.state':state})
    assert replay.index==0
