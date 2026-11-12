import json
from pathlib import Path
import numpy as np
import pytest
from explorer3.datasets import grouped_split
from explorer3.research import features, fit, predict, report, prioritize


def test_split_blocks_transitive_session_or_scene_leakage():
    rows=[{'id':str(i),'session':s,'scene':c} for i,(s,c) in enumerate(
        [('a','1'),('a','2'),('b','2'),('c','3'),('d','4'),('e','5')])]
    split=grouped_split(rows)
    for key, part in split.items():
        other=[r for k,v in split.items() if k!=key for r in v]
        assert not {r['session'] for r in part}&{r['session'] for r in other}
        assert not {r['scene'] for r in part}&{r['scene'] for r in other}
    with pytest.raises(ValueError):
        grouped_split([dict(r,scene='same') for r in rows])


def test_physical_features_do_not_invent_measurements_or_use_future():
    rows=[{'observation_timestamp_ns':i*1000000000,'action_send_start_ns':i*1000000000+1,
           'action_send_end_ns':i*1000000000+2,'action':[i,0], 'observation.state':[i,0],
           'state_valid':True,'measured_response_kind':'previous_command_estimate_no_wheel_feedback',
           'camera':{'timestamp_ns':i*1000000000}} for i in range(1,5)]
    value=features(rows,3000000003,[1,1])
    assert value['response_change'] is None and value['tracking_error'] is None
    assert value==features(rows[:3],3000000003,[1,1])
    rows[2]['action']=[float('nan'),0]
    with pytest.raises(ValueError):features(rows,3000000003,[1,1])


def fixture_rows(prefix,count=48):
    return [{'episode_id':f'{prefix}-{i}','session':f'{prefix}-{i//4}','scene':f'{prefix}-{i//4}',
             'profile':'so101_pick_place' if i%2 else 'tank_dock','assigned':i%2,'propensity':.5,
             'features':{'command_change':float(i%4),'response_change':None,'tracking_error':None,
                         'observation_age_s':.05,'camera_age_s':.1,'send_latency_s':.01,
                         'progress':None,'assigned_delay_s':0.},
             'outcome':'success' if i%4<2 else 'failure','synthetic':True,'events':[],
             'decision_time_ns':0,'emergency':False} for i in range(count)]


def test_estimator_is_assignment_model_with_disjoint_evaluation(tmp_path):
    train=fixture_rows('train')
    artifact=fit(train,tmp_path/'fixture.joblib',synthetic=True)
    result=predict(artifact,fixture_rows('test'))
    assert all(-1<=r['gain']<=1 for r in result)
    assert report(artifact,fixture_rows('test'))['synthetic']
    with pytest.raises(ValueError,match='overlaps'):report(artifact,train)
    with pytest.raises(ValueError,match='ROSMASTER'):
        fit([dict(r,profile='rosmaster_dock') for r in train],tmp_path/'no.joblib',synthetic=True)
    with pytest.raises(ValueError,match='mix synthetic'):fit(train,tmp_path/'no.joblib')


def test_priority_counts_switch_time_and_never_executes_commands():
    items=[dict(id='a',request_time_ns=1,switch_seconds=9,correction_seconds=1,gain=.5,p_without=.2),
           dict(id='b',request_time_ns=2,switch_seconds=0,correction_seconds=1,gain=.2,p_without=.4)]
    assert prioritize(items)[0]['id']=='b'
    assert prioritize(items,'fifo')[0]['id']=='a'
    assert 'priority' not in items[0]
