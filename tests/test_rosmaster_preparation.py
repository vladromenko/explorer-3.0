import json
import pytest
from explorer3 import datasets,rosmaster_preparation as prep


def sources(tmp_path,monkeypatch):
    monkeypatch.setattr(prep,'DATA',tmp_path)
    monkeypatch.setattr(datasets,'DATA',tmp_path)
    entries=[]
    for i in range(2):
        root=tmp_path/'datasets'/str(i);root.mkdir(parents=True)
        (root/'research.jsonl').write_text('{}\n')
        entries.append({'id':str(i),'dataset':str(root),'session':str(i),'scene':str(i),
            'sha256':datasets.fingerprint(root)})
    monkeypatch.setattr(prep,'motion_sources',lambda _: (entries,[]))
    return entries


def test_identification_plan_reopens_and_detects_changed_data(tmp_path,monkeypatch):
    entries=sources(tmp_path,monkeypatch)
    ident,plan=prep.prepare_identification('rosmaster_dock',['0','1'])
    assert prep.load_identification(ident)==plan
    assert not plan['apply_automatically'] and not plan['physical_access']
    assert not (tmp_path/'jobs').exists()
    from pathlib import Path
    (Path(entries[0]['dataset'])/'research.jsonl').write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='Запись изменилась'):prep.load_identification(ident)


def test_identification_cannot_split_one_scene_or_accept_modified_plan(tmp_path,monkeypatch):
    entries=sources(tmp_path,monkeypatch)
    entries[1]['scene']='0'
    with pytest.raises(ValueError,match='независимые'):prep.prepare_identification('rosmaster_dock',['0','1'])
    entries[1]['scene']='1'
    ident,plan=prep.prepare_identification('rosmaster_dock',['0','1'])
    plan['holdout_fraction']=.9
    (tmp_path/'identification-plans'/(ident+'.json')).write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='План идентификации изменён'):prep.load_identification(ident)
