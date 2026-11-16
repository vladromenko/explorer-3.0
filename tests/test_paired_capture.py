import json

import pytest

from explorer3.datasets import grouped_split
from explorer3.sim.paired_capture import KNOWN, CONTROL, control_copy


def test_paired_splits_are_independent_of_episode_uuids():
    known = [dict(id=f'a{i:02}', session=f's{i}', scene=f'c{i}') for i in range(30)]
    control = [dict(e, id=f'z{30-i:02}') for i, e in enumerate(known)]
    a, b = grouped_split(known), grouped_split(control)
    for key in ('train', 'validation', 'test'):
        assert {e['scene'] for e in a[key]} == {e['scene'] for e in b[key]}
    assert len(a['train']) == 24 and len(a['test']) == len(a['validation']) == 3


def test_control_derivative_preserves_pixels_actions_and_pairing(tmp_path):
    source = tmp_path / 'capture'
    source.mkdir()
    meta = dict(profile=KNOWN, domain='sim', task='pick up the cube, 3.2 x 3.2 cm, 3.2 cm high',
                paired_group='group', session='group', scene='group',
                episode_parameters={'object': {'name': 'cube'}})
    (source / 'manifest.json').write_text(json.dumps(meta))
    (source / '000000.png').write_bytes(b'frame')
    (source / 'samples.jsonl').write_text('{"action": [10, 20]}\n')
    target = control_copy(source)
    actual = json.loads((target / 'manifest.json').read_text())
    assert actual['profile'] == CONTROL and actual['task'] == 'pick up the cube'
    assert actual['scene'] == actual['session'] == 'group'
    for name in ('000000.png', 'samples.jsonl'):
        assert (source / name).stat().st_ino == (target / name).stat().st_ino
    assert control_copy(source) == target
    assert json.loads((source / 'manifest.json').read_text()) == meta
    with pytest.raises(ValueError):
        control_copy(target)


def test_donor_lineage_is_grouped_transitively_across_generated_scenes():
    from explorer3.datasets import connected_groups
    episodes = [dict(id='a', session='s1', scene='c1', lineage_sources=['donor-1']),
                dict(id='b', session='s2', scene='c2', lineage_sources=['donor-1', 'donor-2']),
                dict(id='c', session='s3', scene='c3', lineage_sources=['donor-2']),
                dict(id='independent', session='s4', scene='c4')]
    assert {frozenset(e['id'] for e in group) for group in connected_groups(episodes)} == {
        frozenset(('a', 'b', 'c')), frozenset(('independent',))}
    with pytest.raises(ValueError, match='3'):
        grouped_split(episodes)


def test_donor_lineage_is_read_from_immutable_recording_provenance(tmp_path):
    from explorer3.datasets import connected_groups
    rows = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        (root / 'provenance.json').write_text(json.dumps({'hardware': {'transfer_lineage': {
            'source_episodes': [{'id': 'shared-donor', 'sha256': 'a' * 64}]}}}))
        rows.append(dict(id=str(index), session='session-' + str(index), scene='scene-' + str(index), dataset=str(root)))
    assert len(connected_groups(rows)) == 1
