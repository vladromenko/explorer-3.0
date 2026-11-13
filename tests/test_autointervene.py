"""Declared numerical fixtures, never reported as robot/calibration experiments."""
import copy
import numpy as np
import pytest
from explorer3.autointervene import Memory,Settings,SupportMonitor,calibrate


def data():
    memory={'id':'m','session':'memory-session','scene':'memory-scene','outcome':'success','role':'memory',
        'policy_id':'weights-a','profile':'arm','features':[[[1.,0.]],[[.8,.6]]],'actions':[[0.,0.],[10.,2.]]}
    held={**memory,'id':'h','session':'held-session','scene':'held-scene','role':'calibration',
          'proposals':[[[0.,0.]],[[10.,2.]]],'proposal_kind':'policy_prediction','data_kind':'numerical_fixture'}
    settings=Settings(prefix=1,visual_neighbors=2,action_neighbors=1,trajectories=1,window=2,
        risk_history=1,reject_persistence=2,accept_persistence=2,advance_actions=1)
    return memory,held,settings


def test_risk_and_visual_support_use_same_worst_group_neighbors():
    source,_,settings=data();memory=Memory([source],profile='arm',policy_id='weights-a',settings=settings,groups=[[0],[1]])
    result=SupportMonitor(memory).score([[1.,0.]],[[0.,3.]])
    # Std=[5,1]. Group 0 matches first reference (distance 0), group 1
    # matches second (distance 1); its visual support is .8, not 1.
    assert result['risk']==1 and result['support']==pytest.approx(.8) and result['references']==[1]


def test_calibration_rejects_leakage_expert_substitution_and_test():
    source,held,settings=data();memory=Memory([source],profile='arm',policy_id='weights-a',settings=settings)
    for change in ({'session':source['session']},{'scene':source['scene']},{'role':'test'},
                   {'proposal_kind':'recorded_expert_action'},{'policy_id':'weights-b'}):
        with pytest.raises(ValueError):calibrate(memory,[{**held,**change}])
    thresholds=calibrate(memory,[held])
    assert thresholds['calibration_data_kind']==['numerical_fixture']
    monitor=SupportMonitor(memory,thresholds)
    query=dict(features=[[1.,0.]],proposal=[[0.,0.]],profile='arm',policy_id='weights-a')
    assert monitor.assess(**query)['mode']=='operator'
    assert monitor.assess(**query)['mode']=='policy'
    bad={**query,'features':[[-1.,0.]]}
    saved=monitor.snapshot();first=monitor.assess(**bad);second=monitor.assess(**bad)
    assert first['mode']=='policy' and second['mode']=='operator' and second['reset_policy_chunks']
    monitor.restore(saved)
    assert monitor.assess(**bad)==first and monitor.assess(**bad)==second
    with pytest.raises(ValueError):monitor.assess(**{**query,'policy_id':'weights-b'})


def test_memory_roundtrip_and_local_windows_do_not_rewind(tmp_path):
    source,held,settings=data();memory=Memory([source],profile='arm',policy_id='weights-a',settings=settings)
    memory.save(tmp_path/'memory.npz');loaded=Memory.load(tmp_path/'memory.npz')
    assert loaded.ident==memory.ident and np.array_equal(loaded.actions,memory.actions)
    monitor=SupportMonitor(loaded,calibrate(memory,[held]));monitor.reset('policy')
    result=monitor.score([[.8,.6]],[[10.,2.]],executed_actions=1)
    assert result['exhausted'] and monitor.windows[0]==1
    monitor.score([[1.,0.]],[[0.,0.]],executed_actions=1)
    assert monitor.windows[0]==1
    with (tmp_path/'memory.npz').open('ab') as output:output.write(b'changed')
    with pytest.raises(ValueError):Memory.load(tmp_path/'memory.npz')
