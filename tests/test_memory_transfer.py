"""Cross-platform memory scope checks, without model or robot execution."""
import json
import time
import pytest
from explorer3 import catalog
from explorer3.cognitive import registry,store,transfer
from explorer3.cognitive.schemas import ObservationEvidence,AssistanceCase,CompatibilityRecord


def test_cross_platform_transfer_keeps_new_scope_unverified(tmp_path,monkeypatch):
    monkeypatch.setattr(catalog,'DATA',tmp_path);monkeypatch.setattr(store,'DATA',tmp_path)
    catalog.initialize();store.initialize();now=time.time()
    evidence=[]
    for robot in ('tank','rosmaster'):
        payload={'archived':True}
        evidence.append(store.put_evidence(ObservationEvidence(id=robot,domain='sim',robot=robot,source='fixture',sample_id=robot,
            frame_id=robot+'/rgb',received_timestamp_ns=1,clock_domain='fixture',created_at=now,expires_at=now+1,
            status='observed',content_sha256=store.digest(payload),payload=payload)))
    c=registry.get('tank.dock_fixed.scripted.v1')
    scope=CompatibilityRecord(recipient_skill_id=c.skill_id,recipient_version=c.version,policy_sha256=c.policy_sha256,
        camera_sha256=c.camera_sha256,calibration_sha256=None,domain='sim',goal_id='fixed_dock',
        checked_conditions=['fixed_scene'],unknown_conditions=[],evidence_ids=['tank'])
    case=AssistanceCase(id='source',task='dock',effect='target_visible',domain='sim',donor='rosmaster',recipient='tank',
        donor_episode='first',recipient_episode='second',reason='fixture',before_evidence_ids=['tank'],compatibility=scope,
        assistance_type='observation',expected_effect='target visible',actions_provenance='observation',actual_effect={})
    store.add_case(case)
    with pytest.raises(ValueError,match='другую платформу'):transfer.prepare('source',c.skill_id,['tank'])
    with pytest.raises(ValueError,match='собственное наблюдение'):transfer.prepare('source','rosmaster.dock_fixed.scripted.v1',['tank'])
    value=transfer.prepare('source','rosmaster.dock_fixed.scripted.v1',['rosmaster'])
    assert value['donor']=='tank' and value['recipient']=='rosmaster' and value['donor_episode']=='second'
    assert value['source_scope_mismatches'] and not value['fresh_recipient_observation']
    assert not value['execution_started'] and not value['recipient_weights_updated']
    transfer.validate(value)
    with catalog.connect() as db:
        changed=case.model_copy(update={'status':'contradicted'}).model_dump()
        db.execute('UPDATE cognitive_cases SET body=? WHERE id=?',(json.dumps(changed),'source'))
    with pytest.raises(ValueError,match='changed'):transfer.validate(value)
