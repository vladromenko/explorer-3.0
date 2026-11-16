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
