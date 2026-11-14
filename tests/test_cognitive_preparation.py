"""Prepared regression cases; not executed under the operator's launch freeze."""
import hashlib
import pytest
from explorer3.cognitive import protocol,shadow,store
from explorer3.cognitive.schemas import ObservationEvidence


def test_memory_does_not_change_actions_or_instruction():
    memory=[{'id':'case','actual_effect':{'direct_pose_verified':True}}]
    requests=[protocol.decision_request(mode,[],memory,'a'*64) for mode in ('M0','M1','M2','M3')]
    assert all(r['allowed_choices']==list(protocol.CHOICES) for r in requests)
    assert all(r['instruction']==requests[0]['instruction'] for r in requests)
    assert [r['memory'] for r in requests]==[[],[],memory,memory]
    with pytest.raises(ValueError):protocol.decision_request('unknown',[],[],None)


def evidence(**changes):
    payload={'archived':True,'streams':{key:{'age_s':.1} for key in shadow.STREAMS}}
    values=dict(id='archive',domain='real',robot='rosmaster',source='recorded_sensor_snapshot',
        sample_id='sample',frame_id='rosmaster/local_sensor_frames',received_timestamp_ns=1,
        clock_domain='rosmaster_monotonic',created_at=10,expires_at=12,status='observed',
        content_sha256=store.digest(payload),payload=payload)
    values.update(changes)
    return ObservationEvidence(**values)


def test_archive_never_grants_readiness_or_outcomes():
    result=shadow.summarize(evidence(),now=1000)
    assert all(r['capture_status']=='reported_recent_at_capture' for r in result['streams'])
    assert result['live_observation'] is False and result['motion_admitted'] is False
    assert result['hardware_readiness']==result['task_outcome']=='unknown'
    assert result['model_called'] is False


def test_shadow_rejects_wrong_domain_robot_or_modified_payload():
    for values in ({'domain':'sim'},{'robot':'tank'},{'content_sha256':'0'*64}):
        with pytest.raises(ValueError):shadow.summarize(evidence(**values),now=1000)


def test_missing_and_negative_age_not_reported_as_recent():
    item=evidence();payload={**item.payload,'streams':{'rgb':{'age_s':-1},'depth':{'age_s':True}}}
    result=shadow.summarize(item.model_copy(update={'payload':payload,'content_sha256':store.digest(payload)}),now=1000)
    assert result['proposed_next_step']=='review_missing_streams'
    assert all(r['capture_status']!='reported_recent_at_capture' for r in result['streams'])


def test_archive_review_rechecks_file_before_saving_report(tmp_path,monkeypatch):
    packet=tmp_path/'packet.npz';packet.write_bytes(b'original fixture')
    item=evidence(artifact_ref='packet.npz',artifact_sha256=hashlib.sha256(packet.read_bytes()).hexdigest())
    monkeypatch.setattr(store,'DATA',tmp_path)
    monkeypatch.setattr(store,'evidence',lambda *a,**k:[item])
    events=[];monkeypatch.setattr(store,'event',lambda *args:events.append(args))
    shadow.review(item.id)
    assert len(events)==1
    packet.write_bytes(b'modified fixture')
    with pytest.raises(ValueError,match='artifact hash'):shadow.review(item.id)
    assert len(events)==1
