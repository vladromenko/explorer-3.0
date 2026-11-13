"""Fixture contracts only: no datasets written as demonstrations and no training."""
import copy,json
import pytest
from explorer3 import transfer_preparation as transfer,training_drafts as drafts,datasets
from explorer3.profiles import profile


def source(tmp_path):
    cfg=profile('rosmaster_dock');cp=tmp_path/'checkpoint';cp.mkdir()
    (cp/'config.json').write_text(json.dumps({'type':'act','input_features':{
        'observation.state':{'shape':[6]},'observation.images.front':{'shape':[3,480,640]}},'output_features':{'action':{'shape':[3]}}}))
    (cp/'model.safetensors').write_bytes(b'explicit unit fixture, not a policy')
    plan={'domain':'sim','profile':'rosmaster_dock_sim','profile_contract':profile('rosmaster_dock_sim'),'smoke':True,'sim_model_hash':'a'*64}
    job={'id':'source','profile':'rosmaster_dock_sim','policy':'act','status':'completed','checkpoint':str(cp),'config':json.dumps({'plan_id':'s'*64})}
    return job,plan


def test_prepare_equal_real_data_budget_and_immutable_checkpoint(tmp_path,monkeypatch):
    job,old=source(tmp_path)
    new={'domain':'real','profile':'rosmaster_dock','splits':{r:[{'id':r,'frames':20}] for r in ('train','validation','test')}}
    monkeypatch.setattr(transfer,'DATA',tmp_path);monkeypatch.setattr(drafts,'DATA',tmp_path)
    monkeypatch.setattr(datasets,'load_plan',lambda ident:old if ident=='s'*64 else new)
    def freeze(name,smoke,**kw):
        assert name=='rosmaster_dock' and smoke is False and kw['min_real_episodes']==3 and kw['selected_ids']==['a','b','c']
        return 'r'*64,new
    monkeypatch.setattr(datasets,'freeze',freeze)
    ident,pair=transfer.prepare(job,'rosmaster_dock',['a','b','c'],steps=2000,batch=1,seed=7,device='cpu')
    assert transfer.load_comparison(ident)==pair
    a=drafts.load(pair['transfer_draft']);b=drafts.load(pair['scratch_draft'])
    for key in ('plan_id','steps','batch_size','seed','device','frames'):assert a[key]==b[key]
    assert a['mixture']['sim_train_frames']==0 and not a['training_started']
    assert a['initial_checkpoint'] and b['initial_checkpoint'] is None
    assert not (tmp_path/'jobs').exists()
    from pathlib import Path
    (Path(job['checkpoint'])/'processor.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='веса изменились'):drafts.load(pair['transfer_draft'])


def test_semantic_channel_units_and_camera_contract_not_only_dimensions(tmp_path,monkeypatch):
    job,plan=source(tmp_path)
    monkeypatch.setattr(datasets,'load_plan',lambda _:plan)
    assert transfer.check_source(job,'rosmaster_dock')
    plan['profile_contract']=copy.deepcopy(plan['profile_contract']);plan['profile_contract']['units'][0]='cm/s'
    with pytest.raises(ValueError,match='единицы'):transfer.check_source(job,'rosmaster_dock')
    with pytest.raises(ValueError,match='другой задаче'):transfer.check_source(job,'rosmaster_pick_place')


def test_diagnostic_clip_cannot_enter_real_training_even_with_success(tmp_path,monkeypatch):
    root=tmp_path/'clip';root.mkdir();(root/'provenance.json').write_text(json.dumps({'domain':'real','synthetic':False,'hardware':{'pipeline_check':True}}))
    entries=[{'id':str(i),'dataset':str(root),'profile':'rosmaster_dock'} for i in range(3)]
    monkeypatch.setattr(datasets,'eligible',lambda _:entries)
    with pytest.raises(ValueError,match='Диагностическая'):datasets.freeze('rosmaster_dock',False,min_real_episodes=3)
