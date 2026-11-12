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


def test_local_mode_blocks_trial_before_loading_hardware(monkeypatch):
    monkeypatch.setattr(trials.local_work,'enabled',lambda:True)
    with pytest.raises(PermissionError,match='локальный режим'):
        trials.require_physical({'profile':'so101_pick_place','id':'0'*32})


def test_recording_rejects_source_switch_before_writing_frame():
    writer=object.__new__(EpisodeWriter)
    writer.camera='wrist';writer.category='rollout';writer.command_source='policy'
    sample={'images':{'wrist':{}},'command_source':'human'}
    with pytest.raises(ValueError,match='segment'):
        writer.add(sample)
