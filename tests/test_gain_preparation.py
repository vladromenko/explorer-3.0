import copy
import json
import math
import pytest
from explorer3 import gain_preparation as gp,gain_model
from explorer3.research import FEATURES,matrix


def report(seed):
    return {'domain':'sim','replay_verified':True,'policy':{'kind':'frozen_policy','policy_id':'frozen-a'},
        'decision_context':{'features':{name:(.1 if name=='assigned_delay_s' else None) for name in FEATURES},
            'source':'controller_visible_history','truth_features':False},'lineage':str(seed),
        'outcomes':[{'branch':b,'repeat':0,'outcome':'success' if seed%2 else 'failure'} for b in ('continue','hold_only','correction')],
        'budget':{'profile':'tank_dock_sim','seed':seed},'model':{'xml_sha256':'same-world'},
        'actual_windows':{'correction_s':1.,'horizon_s':3.,'delay_s':.1},'reset_rule':'matched reset'}


def test_prepare_and_reload_are_immutable_do_not_fit_and_reject_censoring(tmp_path,monkeypatch):
    monkeypatch.setattr(gp,'DATA',tmp_path)
    import sklearn.linear_model
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression,'fit',lambda *_args,**_kwargs:pytest.fail('training forbidden'))
    paths=[]
    for i in range(20):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps(report(i)));paths.append(p)
    ident,plan=gp.prepare(paths)
    assert not plan['training_started'] and gp.load(ident)==plan
    roles={}
    for row in plan['rows']:roles.setdefault(row['scene_group'],set()).add(row['split'])
    assert all(len(r)==1 for r in roles.values())
    assert set(r['split'] for r in plan['rows'])=={'train','validation','test'}
    assert gp.prepare(paths)[0]==ident
    paths[0].write_text('{}')
    with pytest.raises(ValueError,match='report changed'):gp.load(ident)
    r=report(20);r['outcomes'][0]['outcome']='censored'
    with pytest.raises(ValueError,match='Незавершённые'):gp.candidate(r)
    r=report(21);r['policy']['kind']='diagnostic_hold'
    with pytest.raises(ValueError,match='удержание'):gp.candidate(r)
    r=report(22);r['decision_context']['features']['hidden_truth']=1
    with pytest.raises(ValueError,match='признаков'):gp.candidate(r)


def test_same_scene_across_delays_cannot_leak(tmp_path,monkeypatch):
    monkeypatch.setattr(gp,'DATA',tmp_path);paths=[]
    for i in range(20):
        for delay in (0.,.2):
            r=report(i);r['lineage']=f'{i}-{delay}';r['actual_windows']['delay_s']=delay
            r['decision_context']['features']['assigned_delay_s']=delay
            p=tmp_path/f'{i}-{delay}.json';p.write_text(json.dumps(r));paths.append(p)
    _,plan=gp.prepare(paths)
    for scene in {r['scene_group'] for r in plan['rows']}:
        assert len({r['split'] for r in plan['rows'] if r['scene_group']==scene})==1


def test_visible_history_masks_tank_response_and_future_commands():
    rows=[gp.sample('tank_dock_sim',[0,0],[.1,0],t) for t in (0.,.05)]
    feature=gp.observation(rows,.1,[.4,1.],.2)
    assert feature['response_change'] is None and feature['tracking_error'] is None
    assert feature['observation_age_s']==pytest.approx(.05)
    assert gp.observation(rows+[gp.sample('tank_dock_sim',[999,999],[999,999],9)],.1,[.4,1.],.2)==feature
    arm=[gp.sample('so101_pick_place_sim',[i]*6,[.5]*6,i*.05) for i in (0,1)]
    assert gp.observation(arm,.1,[2]*6,0)['response_change']==pytest.approx(10.)


def test_json_inference_uses_masks_and_rejects_invalid_weights():
    rows=[{'features':{n:None for n in FEATURES}}];_,names=matrix(rows)
    head={'mean':[0.]*len(names),'scale':[1.]*len(names),'coef':[0.]*len(names),'intercept':0.}
    artifact={'include_physical':True,'feature_names':names,'heads':[head,{**head,'intercept':math.log(3.)}]}
    assert gain_model.predict(artifact,rows)==[{'p_without':.5,'p_with':.75,'gain':.25}]
    broken=copy.deepcopy(artifact);broken['heads'][0]['scale'][0]=0
    with pytest.raises(ValueError,match='weights'):gain_model.predict(broken,rows)
