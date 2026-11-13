import json
import pytest
from explorer3 import fleet_corrections as module


def test_review_pins_source_snapshot_and_success(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    root=tmp_path/'staging/sim/segment';root.mkdir(parents=True)
    (root/'entry-snapshot.json').write_text('{}')
    (root/'samples.jsonl').write_text('{}\n')
    (root/'000000.png').write_bytes(b'frame')
    meta={'domain':'sim','command_source':'scripted_expert','segment_role':'fleet_correction',
          'selection_origin':'frozen_checkpoint','outcome':'success','session':'run','lineage':'run',
          'entry_snapshot_sha256':module.digest(root/'entry-snapshot.json')}
    (root/'manifest.json').write_text(json.dumps(meta))
    review={'decision':'accept_fleet_sim_correction','source_sha256':module.source_hash(root),'lineage':'run'}
    assert module.validate(root,review)==meta
    with pytest.raises(ValueError,match='проверки'):module.validate(root,{})
    (root/'000000.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='проверки'):module.validate(root,review)
    (root/'entry-snapshot.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='снимок'):module.validate(root)
    meta['outcome']='aborted';(root/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='успехом'):module.validate(root)
    meta['selection_origin']='numerical_fixture';(root/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Программный пример'):module.validate(root)
