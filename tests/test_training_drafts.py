import json
from pathlib import Path
import pytest
from explorer3 import training_drafts as drafts


def episode(ident,session=None,scene=None):
    return {'id':ident,'session':session or ident,'scene':scene or ident,'sha256':ident*64,'frames':100}


def plan(train,validation,test):
    return {'domain':'sim','profile':'tank_dock_sim','sim_model_hash':'a'*64,
        'splits':{'train':train,'validation':validation,'test':test},'smoke':False,'seed':1000}


def test_finetune_preserves_old_roles_and_counts_new_frames():
    old=plan([episode('a')],[episode('b')],[episode('c')])
    new=plan([episode('d')],[episode('e',scene='a')],[episode('f')])
    merged=drafts.merge_plans(old,new)
    assert {x['id'] for x in merged['splits']['train']}=={'a','d','e'}
    assert {x['id'] for x in merged['splits']['validation']}=={'b'}
    assert {x['id'] for x in merged['splits']['test']}=={'c','f'}
    # A new segment bridges old train and validation: reject rather than reassign.
    new=plan([episode('d',session='a',scene='b')],[episode('e')],[episode('f')])
    with pytest.raises(ValueError,match='связывают'):drafts.merge_plans(old,new)


def test_preparation_creates_only_draft_and_frozen_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(drafts,'DATA',tmp_path)
    selected=plan([episode('a')],[episode('b')],[episode('c')])
    monkeypatch.setattr(drafts.datasets,'freeze',lambda *a,**kw:('p'*64,selected))
    monkeypatch.setattr(drafts.datasets,'load_plan',lambda ident:selected)
    ident,prepared=drafts.prepare('tank_dock_sim','act',['a','b','c'],steps=1234,batch=2,seed=1000,device='cpu')
    assert prepared['preparation_only'] and not prepared['training_started']
    assert prepared['steps']==1234 and prepared['mixture']=={'old_train_frames':0,'new_train_frames':100}
    assert drafts.load(ident)==prepared
    assert not (tmp_path/'jobs').exists() and not (tmp_path/'catalog.sqlite3').exists()
    (tmp_path/'training-drafts'/(ident+'.json')).write_text('{}')
    with pytest.raises(ValueError,match='изменился'):drafts.load(ident)


def test_finetune_checkpoint_is_pinned_and_changed_weights_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(drafts,'DATA',tmp_path)
    old=plan([episode('a')],[episode('b')],[episode('c')]);new=plan([episode('d')],[episode('e')],[episode('f')])
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps({'type':'act'}));(checkpoint/'model.safetensors').write_bytes(b'weights')
    monkeypatch.setattr(drafts.datasets,'freeze',lambda *a,**kw:('n'*64,new))
    monkeypatch.setattr(drafts.datasets,'load_plan',lambda ident:old if ident=='o'*64 else new)
    source={'id':'source','status':'completed','profile':'tank_dock_sim','policy':'act','checkpoint':str(checkpoint),
            'config':json.dumps({'kind':'train','plan_id':'o'*64})}
    ident,value=drafts.prepare('tank_dock_sim','act',['d','e','f'],steps=100,batch=1,seed=1000,device='cpu',source=source)
    assert value['mode']=='fine_tune_new_optimizer' and value['mixture']=={'old_train_frames':100,'new_train_frames':100}
    drafts.load(ident)
    (checkpoint/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='веса изменились'):drafts.load(ident)


@pytest.mark.parametrize('args',[(49,1,0,'cpu'),(100,0,0,'cpu'),(100,1,-1,'cpu'),(100,1,0,'auto')])
def test_invalid_budgets(args):
    with pytest.raises(ValueError):drafts.validate_budget(*args)
