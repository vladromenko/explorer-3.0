"""Per-episode language must survive dataset freezing without weakening sensor/task contracts."""
import copy
import json

import pytest

from explorer3 import datasets
from explorer3.profiles import profile
from explorer3.training_drafts import semantic_contract


@pytest.mark.parametrize('name', ['rosmaster_floor_pick_knowledge_sim', 'rosmaster_floor_pick_noknowledge_sim'])
def test_parameterised_task_can_be_frozen_with_independent_splits(name, monkeypatch, tmp_path):
    cfg = profile(name)
    monkeypatch.setattr(datasets, 'DATA', tmp_path)
    entries = []
    for i, task in enumerate(('pick up the cube, 3.2 x 3.2 cm, 3.2 cm high',
                               'pick up the bottle, 4.0 x 4.0 cm, 9.0 cm high',
                               'pick up the block, 6.0 x 3.0 cm, 3.0 cm high')):
        if name.endswith('_noknowledge_sim'):
            task = task.split(',')[0]
        root = tmp_path / 'datasets' / str(i)
        root.mkdir(parents=True)
        (root / 'provenance.json').write_text(json.dumps({'domain': 'sim', 'synthetic': True,
            'profile': {**cfg, 'task': task}, 'hardware': {'model': {'xml_sha256': 'a' * 64}}}))
        (root / 'research.jsonl').write_text(json.dumps({'command_source': 'scripted_expert'}) + '\n')
        entry = {'id': str(i), 'session': str(i), 'scene': str(i), 'dataset': str(root), 'frames': 1}
        validation = tmp_path / 'validation'
        validation.mkdir(exist_ok=True)
        # Dataset readback itself has separate integration tests. Use its hash-bound result
        # here to exercise the real freeze/split logic, rather than mocking that logic away.
        (validation / f'episode-{i}.json').write_text(json.dumps({'readback': 'PASS', 'dataset_sha256': datasets.fingerprint(root)}))
        entries.append(entry)
    monkeypatch.setattr(datasets, 'eligible', lambda _: copy.deepcopy(entries))
    ident, plan = datasets.freeze(name, False, min_sim_episodes=3)
    assert {k: len(v) for k, v in plan['splits'].items()} == {'train': 1, 'validation': 1, 'test': 1}
    assert datasets.load_plan(ident) == plan
    altered = tmp_path / 'datasets' / '0' / 'provenance.json'
    p = json.loads(altered.read_text())
    p['profile']['camera'] = 'wrong_camera'
    altered.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='Камера'):
        datasets.freeze(name, False, min_sim_episodes=3)


def test_different_task_families_and_fixed_tasks_remain_incompatible():
    known = profile('rosmaster_floor_pick_knowledge_sim')
    control = profile('rosmaster_floor_pick_noknowledge_sim')
    assert semantic_contract(known) != semantic_contract(control)
    assert semantic_contract(known) != semantic_contract({**known, 'task_variant': 'place_in_box'})
    fixed = profile('rosmaster_floor_pick_sim')
    assert semantic_contract(fixed) != semantic_contract({**fixed, 'task': 'put in a box'})
