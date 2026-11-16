import json
import time

import pytest

from explorer3 import catalog, knowledge, world_memory
from explorer3.cognitive import registry, store
from explorer3.cognitive.gap_advisor import validate as validate_advice


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, 'DATA', tmp_path)
    monkeypatch.setattr(catalog, 'prepare', lambda: None)
    monkeypatch.setattr(knowledge, 'STORE', tmp_path / 'knowledge.json')
    catalog.initialize()
    store.initialize()
    monkeypatch.setattr(world_memory, 'target_for', lambda *a: None)
    return tmp_path


def test_missing_location_skill_and_physical_impossibility_are_distinct(isolated, monkeypatch):
    out = knowledge.assess('rosmaster', 'cube')
    assert out['readiness']['where_missing'] and out['readiness']['skill_missing']
    assert out['gaps'] == ['where', 'how', 'skill']
    assert not out['readiness']['physically_infeasible']
    monkeypatch.setattr(world_memory, 'target_for', lambda *a: {'uncertainty_m': None})
    assert knowledge.assess('rosmaster', 'cube')['readiness']['where_missing']
    monkeypatch.setattr(world_memory, 'target_for', lambda *a: {'uncertainty_m': .1})
    context = {'feasibility': {'reachable': False, 'reason': 'object wider than measured jaw limit', 'evidence_ref': 'fixture-measurement'}}
    out = knowledge.assess('rosmaster', 'cube', context=context)
    assert not out['readiness']['where_missing'] and out['readiness']['physically_infeasible']
    assert out['readiness']['skill_missing']


def test_another_object_and_explicit_baseline_do_not_become_acquired_skill(isolated):
    knowledge.add('cube', 'measurement', {'width_m': .03}, object_instance='cube-one')
    out = knowledge.assess('rosmaster', 'cube', context={'object_instance': 'cube-two'}, executor='geometric_baseline')
    assert out['how'] is None and out['readiness']['skill_missing']
    assert out['baseline']['selected'] and not out['baseline']['learned_skill']
    assert 'skill' not in out['gaps']  # explicit experiment executor, still reports the missing learned skill


def test_llm_cannot_remove_a_required_gap_or_invent_a_helper():
    result = validate_advice({'gaps': [], 'helpers': {'skill': ['drone']}}, ['tank'], required_gaps=['where', 'skill'])
    assert result['gaps'] == ['where', 'skill'] and result['helpers'] == {}


def test_verified_skill_is_pinned_reused_and_rejects_changed_conditions(isolated, monkeypatch):
    from explorer3 import training_drafts
    monkeypatch.setattr(training_drafts, 'checkpoint_hash', lambda path: 'a' * 64)
    monkeypatch.setattr(knowledge, 'load_packet', lambda sha: {'fixture': True})
    config = {'donor_packet_shas': ['b' * 64]}
    with catalog.connect() as db:
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',
                   ('job', 'rosmaster_floor_pick_sim', 'smolvla', 'completed', json.dumps(config), time.time(), None, '/fixture/checkpoint'))
    scope = {'object_class': 'cube', 'camera_sha256': 'c' * 64, 'calibration_sha256': 'd' * 64,
             'surface': 'rigid_box', 'size_bounds_m': [[.025, .04]] * 3}
    ids = []
    for index in range(3):
        ident = f'fixture-{index}'
        payload = {'checkpoint_sha256': 'a' * 64, 'scope_sha256': store.digest(scope),
                   'donor_online': False, 'geometric_correction': False, 'recipient': 'success',
                   'lift_m': .1, 'held_s': 1.5, 'scene_id': f'scene-{index}'}
        now = time.time()
        store.put_evidence({'id': ident, 'domain': 'sim', 'robot': 'rosmaster', 'source': 'fixed_outcome_verifier',
                            'sample_id': ident, 'frame_id': 'fixture-only', 'received_timestamp_ns': 1,
                            'clock_domain': 'fixture', 'created_at': now, 'expires_at': now + 30,
                            'status': 'observed', 'content_sha256': store.digest(payload), 'payload': payload})
        ids.append(ident)
    with pytest.raises(ValueError, match='three independent'):
        registry.register_acquired_pick('job', ['b' * 64], scope, ids[:1], domain='sim')
    record = registry.register_acquired_pick('job', ['b' * 64], scope, ids, domain='sim')
    assert registry.register_acquired_pick('job', ['b' * 64], scope, ids, domain='sim') == record
    context = dict(scope, size_m=[.032] * 3)
    out = knowledge.assess('rosmaster', 'cube', 'sim', context=context)
    assert not out['readiness']['skill_missing'] and 'skill' not in out['gaps']
    assert out['registered_skill']['job_id'] == 'job'
    assert registry.get(record['skill_id']).real_admission == 'blocked'
    context['size_m'] = [.07] * 3
    assert knowledge.assess('rosmaster', 'cube', 'sim', context=context)['readiness']['skill_uncertain']
    context['size_m'] = [.032] * 3
    monkeypatch.setattr(training_drafts, 'checkpoint_hash', lambda path: 'e' * 64)
    assert knowledge.assess('rosmaster', 'cube', 'sim', context=context)['readiness']['skill_uncertain']
    assert knowledge.assess('rosmaster', 'cube', 'real', context=context)['readiness']['skill_missing']
    assert len(catalog.rows('jobs')) == 1  # repeated lookup/registration never creates training
