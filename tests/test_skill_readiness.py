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
    assert registry.cards()[record['skill_id']].kind == 'learned'
    context['size_m'] = [.07] * 3
    assert knowledge.assess('rosmaster', 'cube', 'sim', context=context)['readiness']['skill_uncertain']
    context['size_m'] = [.032] * 3
    monkeypatch.setattr(training_drafts, 'checkpoint_hash', lambda path: 'e' * 64)
    assert knowledge.assess('rosmaster', 'cube', 'sim', context=context)['readiness']['skill_uncertain']
    assert knowledge.assess('rosmaster', 'cube', 'real', context=context)['readiness']['skill_missing']
    assert len(catalog.rows('jobs')) == 1  # repeated lookup/registration never creates training


def test_pending_acquisition_survives_restart_and_is_not_duplicate_training(isolated):
    from explorer3.cognitive.broker import prepare_acquisition, acquisition_requests, cancel_acquisition
    assessed = knowledge.assess('rosmaster', 'cube')
    first = prepare_acquisition('mission-1', 'принеси кубик', 'rosmaster', 'cube', assessed, domain='real')
    second = prepare_acquisition('mission-2', 'принеси кубик', 'rosmaster', 'cube', assessed, domain='real',
                                 context={'scene': 'refreshed', 'terminal_goal': {'kind': 'floor_zone'}})
    assert first['id'] == second['id'] and first['state'] == 'needs_demonstration'
    assert not first['training_started'] and not first['motion_authorized']
    assert catalog.rows('jobs') == []
    store.recover_interrupted()
    assert acquisition_requests()[0]['id'] == first['id']
    assert store.mission('mission-2')['auto_resume_motion'] is False
    assert store.mission('mission-2')['terminal_goal'] == {'kind': 'floor_zone'}
    cancel_acquisition(first['id'])
    assert acquisition_requests()[0]['state'] == 'cancelled'
    assert store.mission('mission-1')['phase'] == 'cancelled'
    assert store.mission('mission-2')['phase'] == 'cancelled'


def test_infeasibility_cannot_be_overridden_by_a_stored_skill(isolated):
    from explorer3.cognitive.broker import prepare_acquisition
    assessed = {'registered_skill': {'state': 'available', 'skill_id': 'fixture'},
                'readiness': {'physically_infeasible': True},
                'reasons': {'physically_infeasible': 'Новая сцена недостижима'}}
    request = prepare_acquisition('mission', 'принеси кубик', 'rosmaster', 'cube', assessed, domain='real')
    assert request['state'] == 'rejected' and request['jobs'] == []


def test_transfer_draft_uses_existing_queue_and_cancel_blocks_late_start(isolated, monkeypatch):
    from explorer3 import datasets, training_drafts
    from explorer3.jobs import Jobs
    from explorer3.cognitive import broker
    from test_skill_packets import packet_fixture
    packet = packet_fixture()
    packet['donor']['usage']['allowed'].append('training')
    sha = store.digest(packet)
    monkeypatch.setattr(knowledge, '_load', lambda: [{'object_class': 'cube', 'domain': 'real', 'skill_packet_sha256': sha}])
    monkeypatch.setattr(knowledge, 'load_packet', lambda value: packet)
    monkeypatch.setattr(training_drafts, 'DATA', isolated)
    entries = [{'id': str(i), 'sha256': str(i) * 64, 'frames': 10} for i in range(3)]
    plan = {'domain': 'sim', 'profile': 'rosmaster_floor_pick_sim',
            'splits': dict(zip(('train', 'validation', 'test'), [[e] for e in entries]))}
    monkeypatch.setattr(datasets, 'freeze', lambda *a, **kw: ('p' * 64, plan))
    monkeypatch.setattr(datasets, 'load_plan', lambda *a: plan)
    req = broker.prepare_acquisition('one', 'принеси кубик', 'rosmaster', 'cube', {}, domain='real')
    req = broker.select_acquisition_packet(req['id'], sha)
    with pytest.raises(ValueError, match='вне проверенного переноса'):
        broker.acquisition_training_lineage(req['id'], plan)
    # Queue/state fixture only; actual imported-report validation has a separate run artifact.
    req['transfer_attempts'] = [{'episode_id': e['id'], 'dataset_sha256': e['sha256'], 'accepted': True, 'packet_sha256': sha} for e in entries]
    with catalog.connect() as db:
        broker._save_acquisition(db, req, 'test_fixture')
    draft_id, draft = training_drafts.prepare('rosmaster_floor_pick_sim', 'act', ['0', '1', '2'],
        steps=50, batch=1, seed=17, device='cpu', acquisition_request=req['id'])
    assert draft['donor_packet_shas'] == [sha]
    assert broker.acquisition(req['id'])['state'] == 'awaiting_training_budget'
    jobs = Jobs(None)
    job_id = jobs.start_prepared_training(draft_id)
    assert jobs.start_prepared_training(draft_id) == job_id
    assert len(catalog.rows('jobs')) == 1
    assert broker.acquisition(req['id'])['jobs'] == [job_id]
    with catalog.connect() as db:
        db.execute("UPDATE jobs SET status='completed' WHERE id=?", (job_id,))
    assert broker.reconcile_acquisition(req['id'])['state'] == 'validation'
    assert store.mission('one')['auto_resume_motion'] is False
    with pytest.raises(ValueError, match='Запрос отменён|изменился'):
        jobs.start_prepared_training(draft_id)
    assert len(catalog.rows('jobs')) == 1  # completed acquisition is not silently retrained
    second = broker.prepare_acquisition('two', 'принеси кубик', 'rosmaster', 'cube', {}, domain='real', context={'object_instance': 'different-instance'})
    second = broker.select_acquisition_packet(second['id'], sha)
    second['transfer_attempts'] = req['transfer_attempts']
    with catalog.connect() as db:
        broker._save_acquisition(db, second, 'test_fixture')
    second_draft, _ = training_drafts.prepare('rosmaster_floor_pick_sim', 'act', ['0', '1', '2'],
        steps=50, batch=1, seed=17, device='cpu', acquisition_request=second['id'])
    broker.cancel_acquisition(second['id'])
    with pytest.raises(ValueError, match='Запрос отменён'):
        jobs.start_prepared_training(second_draft)
    assert len(catalog.rows('jobs')) == 1
