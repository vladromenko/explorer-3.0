"""Explicit budget/split checks only, no physical clips or training are fabricated."""
import copy

import pytest

from explorer3 import datasets, training_drafts


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(datasets,'DATA',tmp_path)
    monkeypatch.setattr(training_drafts,'DATA',tmp_path)
    rows=[{'id':str(i),'session':'s'+str(i),'scene':'c'+str(i),'frames':20,'sha256':str(i)*64} for i in range(7)]
    def freeze(name,smoke,**kw):
        assert name=='rosmaster_pick_place' and smoke is False and kw['min_real_episodes']==3
        assert set(kw['selected_ids'])==set(str(i) for i in range(7))
        return 'unused',{'domain':'real','profile':name,'statistics':'train_only',
            'splits':{'train':rows[:4],'validation':rows[4:6],'test':rows[6:]},'sim_model_hash':None}
    monkeypatch.setattr(datasets,'freeze',freeze)
    return {'train':['0','1','2','3','4'],'validation':['5'],'test':['6']},rows


def test_exact_five_training_examples_do_not_inherit_standard_fifty_or_holdouts(prepared):
    partitions,_=prepared
    _,plan=datasets.freeze_target_budget('rosmaster_pick_place',partitions,5)
    assert [e['id'] for e in plan['splits']['train']]==partitions['train']
    assert plan['statistics']=='train_only'
    assert plan['target_data_budget']['k']==5
    assert plan['target_data_budget']['total_target_demonstrations']==7
    assert plan['target_data_budget']['validation_demonstrations']==plan['target_data_budget']['test_demonstrations']==1


def test_connected_donor_derivatives_cannot_cross_explicit_low_data_split(prepared):
    partitions,rows=prepared
    rows[0]['lineage_sources']=['source-a']
    rows[6]['lineage_sources']=['source-a']
    with pytest.raises(ValueError,match='lineages'):
        datasets.freeze_target_budget('rosmaster_pick_place',partitions,5)


def test_claimed_k_missing_holdout_and_duplicate_are_rejected(prepared):
    partitions,_=prepared
    with pytest.raises(ValueError,match='Exactly k'):
        datasets.freeze_target_budget('rosmaster_pick_place',partitions,10)
    bad=copy.deepcopy(partitions);bad['test']=['0']
    with pytest.raises(ValueError,match='Repeated'):
        datasets.freeze_target_budget('rosmaster_pick_place',bad,5)
    with pytest.raises(ValueError,match='supports'):
        datasets.freeze_target_budget('rosmaster_pick_place',partitions,0)


def test_transfer_draft_keeps_exact_target_budget_and_donor_weight_lineage(prepared,monkeypatch,tmp_path):
    import json
    from explorer3 import transfer_preparation as transfer
    partitions,_=prepared
    monkeypatch.setattr(transfer,'DATA',tmp_path)
    source={'id':'donor-pretraining','profile':'rosmaster_floor_pick_sim','policy':'smolvla',
            'checkpoint':'/fixture/pretrained','config':json.dumps({'plan_id':'sim-plan','donor_packet_shas':['a'*64]})}
    monkeypatch.setattr(transfer,'check_source',lambda *a:({'smoke':False,'sim_model_hash':'b'*64},{},'c'*64))
    identifiers=[i for group in partitions.values() for i in group]
    _,comparison=transfer.prepare(source,'rosmaster_pick_place',identifiers,steps=50,batch=1,seed=1000,
                                  device='cpu',partition_ids=partitions,k=5)
    directory=tmp_path/'training-drafts'
    transferred=json.loads((directory/(comparison['transfer_draft']+'.json')).read_text())
    scratch=json.loads((directory/(comparison['scratch_draft']+'.json')).read_text())
    assert transferred['target_data_budget']==scratch['target_data_budget']
    assert transferred['split_counts']=={'train':5,'validation':1,'test':1}
    assert transferred['donor_packet_shas']==['a'*64] and not scratch.get('donor_packet_shas')
    assert not transferred['training_started'] and transferred['initial_checkpoint']
    assert scratch['initial_checkpoint'] is None
    assert not (tmp_path/'jobs').exists()
