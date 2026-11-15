"""«Допуск к роботу»: the checks that decide whether a checkpoint may drive a real arm."""
import json
import types

import pytest

from explorer3 import deploy_gate


class _F:
    def __init__(self, *shape):
        self.shape = shape


def _core(state=6, action=6, cams=('observation.images.wrist',)):
    feats = {'observation.state': _F(state), **{c: _F(3, 480, 640) for c in cams}}
    return types.SimpleNamespace(cfg=types.SimpleNamespace(input_features=feats, output_features={'action': _F(action)}),
                                 image_keys=list(cams))


def test_contract_matches_robot_or_says_what_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_gate, 'DATA', tmp_path)
    assert deploy_gate.check_contract(_core(), 'so101_pick_place', 'j')['status'] == 'pass'
    bad = deploy_gate.check_contract(_core(state=5, cams=('observation.images.front',)), 'so101_pick_place', 'j')
    assert bad['status'] == 'fail' and 'состояния' in bad['message'] and 'wrist' in bad['message']


def test_sim_profile_is_checked_against_its_robot_task():
    assert deploy_gate.real_profile_of('so101_handover_sim') == 'so101_pick_place'
    assert deploy_gate.real_profile_of('rosmaster_pick_place') == 'rosmaster_pick_place'


def test_trial_needs_a_passed_gate_for_the_same_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_gate, 'GATES', tmp_path)
    import explorer3.training_drafts as td
    monkeypatch.setattr(td, 'checkpoint_hash', lambda path: 'h-' + path)
    plan = {'source_id': 'job1', 'checkpoint': 'ck'}
    with pytest.raises(PermissionError, match='не проверена'):
        deploy_gate.require(plan)
    (tmp_path / 'job1.json').write_text(json.dumps({'checkpoint_bundle_sha256': 'h-ck', 'verdict': 'blocked', 'meaning': 'Не допущена: x'}))
    with pytest.raises(PermissionError, match='Не допущена'):
        deploy_gate.require(plan)
    (tmp_path / 'job1.json').write_text(json.dumps({'checkpoint_bundle_sha256': 'h-old', 'verdict': 'ready', 'meaning': ''}))
    with pytest.raises(PermissionError, match='изменились'):
        deploy_gate.require(plan)
    (tmp_path / 'job1.json').write_text(json.dumps({'checkpoint_bundle_sha256': 'h-ck', 'verdict': 'ready', 'meaning': ''}))
    assert deploy_gate.require(plan)['verdict'] == 'ready'


def test_holdout_excludes_training_episodes(monkeypatch):
    rows = [{'id': i, 'profile': 'so101_pick_place', 'outcome': 'success', 'excluded': 0, 'category': 'demonstration',
             'session': 'old' if i == 'd' else 's'} for i in 'abcde']
    monkeypatch.setattr(deploy_gate.catalog, 'rows', lambda table: rows)
    from explorer3 import datasets
    monkeypatch.setattr(datasets, 'series_text', lambda: {'s': 'передача: кубик hand over the cube', 'old': 'кубик в коробку'})
    monkeypatch.setattr(deploy_gate, 'training_ids', lambda job: {'a', 'b'})
    # training episodes and the old «into the box» task are not the hand-over test set
    assert [e['id'] for e in deploy_gate.holdout_episodes({}, 'so101_pick_place')] == ['c', 'e']


def test_cotraining_adds_sim_only_to_train_and_caps_frames(monkeypatch, tmp_path):
    from explorer3 import datasets
    sims = []
    for i in range(5):
        d = tmp_path / f's{i}'
        d.mkdir()
        (d / 'research.jsonl').write_text(json.dumps({'command_source': 'scripted_expert'}) + '\n')
        sims.append({'id': f's{i}', 'dataset': str(d), 'frames': 100, 'created': i, 'profile': 'so101_handover_sim'})
    monkeypatch.setattr(datasets, 'eligible', lambda name: sims if name == 'so101_handover_sim' else [])
    monkeypatch.setattr(datasets, 'simulation_model_hash', lambda e: 'm')
    monkeypatch.setattr(datasets, 'fingerprint', lambda p: 'sha')
    chosen, info = datasets.cotraining_episodes('so101_pick_place', 250, 1.0)
    assert len(chosen) == 2 and info['sim_frames'] == 200 and all(e['domain'] == 'sim' for e in chosen)
    assert info['splits_with_sim'] == ['train']
    with pytest.raises(ValueError):
        datasets.cotraining_episodes('tank_dock', 100, 1.0)
