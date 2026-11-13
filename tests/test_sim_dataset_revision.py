import json
import pytest
from explorer3.datasets import select_simulation_revision


def test_changed_physics_cannot_silently_mix_with_old_geometry(tmp_path):
    episodes=[]
    for ident,version in [('old','a'*64),('new','b'*64)]:
        root=tmp_path/ident;root.mkdir()
        (root/'provenance.json').write_text(json.dumps({'hardware':{'model':{'xml_sha256':version}}}))
        episodes.append({'id':ident,'dataset':str(root)})
    with pytest.raises(ValueError,match='разным версиям'):select_simulation_revision(episodes)
    selected,version=select_simulation_revision(episodes,'b'*64)
    assert [e['id'] for e in selected]==['new'] and version=='b'*64
    assert select_simulation_revision(episodes,'c'*64)==([],'c'*64)
