import json
import pytest
from explorer3 import branch_corrections as module


def test_only_reviewed_unchanged_successful_sim_segment_is_admitted(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'DATA',tmp_path)
    root=tmp_path/'staging/sim/segment';root.mkdir(parents=True)
    meta={'domain':'sim','command_source':'scripted_expert','segment_role':'bounded_correction',
          'lineage':'same-group','session':'same-group','snapshot_sha256':'same-state','outcome':'success'}
    (root/'manifest.json').write_text(json.dumps(meta));(root/'samples.jsonl').write_text('{}\n');(root/'000000.png').write_bytes(b'frame')
    report={'domain':'sim','replay_verified':True,'lineage':'same-group','snapshot_sha256':'same-state',
        'outcomes':[{'branch':'correction','repeat':0,'outcome':'success'}],
        'correction_candidates':[{'spool':str(root),'frames':1}]}
    assert module.candidate(report)[0]==root
    with pytest.raises(ValueError):module.validate_review(root,{})
    review={'decision':'accept_sim_scripted_correction','source_sha256':module.source_hash(root),
        'lineage':'same-group','snapshot_sha256':'same-state'}
    module.validate_review(root,review)
    (root/'000000.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='источник изменился'):module.validate_review(root,review)
    report['outcomes'][0]['outcome']='censored'
    with pytest.raises(ValueError,match='успехом'):module.candidate(report)
