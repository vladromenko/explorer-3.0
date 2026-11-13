"""Finite branch protocol checks; no optimizer and no physical transport."""
import json
import numpy as np
import pytest
from explorer3.sim import state_codec
from explorer3.sim.branching import run,validate_config


def config(**kwargs):
    return dict(profile='tank_dock_sim',seed=14,warmup_s=.15,delay_s=.1,
                correction_s=.3,horizon_s=.6,repeats=2,**kwargs)


def test_typed_snapshot_roundtrip_is_stable(tmp_path):
    value={'array':np.arange(6,dtype=np.float64).reshape(2,3),'tuple':(2,[None,True]),'scalar':np.int64(9)}
    path=tmp_path/'state.json';digest=state_codec.dump(path,value)
    restored=state_codec.load(path)
    np.testing.assert_array_equal(restored['array'],value['array'])
    assert isinstance(restored['tuple'],tuple) and restored['tuple']==value['tuple']
    assert state_codec.dump(path,restored)==digest
    with pytest.raises(ValueError):state_codec.encode(np.array([object()]))


def test_native_three_branches_restore_and_match_intervention_window(tmp_path):
    result=run(config(),tmp_path)
    assert result['replay_verified'] and len(result['outcomes'])==6
    assert len({r['initial_snapshot_sha256'] for r in result['outcomes']})==1
    assert len({r['split_group'] for r in result['outcomes']})==1
    first={r['branch']:r for r in result['outcomes'] if r['repeat']==0}
    assert first['continue']['reset_step'] is None
    assert first['hold_only']['reset_step']==first['correction']['reset_step']==8
    assert first['hold_only']['matched_hold_sim_s']==first['correction']['correction_sim_s']
    assert not result['training_started'] and not result['human_effect_verified']
    traces={k:[json.loads(x) for x in (tmp_path/v['trace']).read_text().splitlines()] for k,v in first.items()}
    for a,b in zip(traces['hold_only'][:2],traces['correction'][:2]):
        assert a['action']==b['action'] and a['physics_qpos']==b['physics_qpos']
    assert any(abs(x)>0 for x in traces['correction'][2]['action'])
    assert first['continue']['outcome']=='censored'
    with pytest.raises(ValueError,match='already contains'):run(config(),tmp_path)


def test_invalid_budgets_and_wrong_model_rejected(tmp_path):
    for key,value in [('delay_s',float('nan')),('repeats',1),('correction_s',99)]:
        cfg=config();cfg[key]=value
        with pytest.raises(ValueError):validate_config(cfg)
    cfg=config();cfg.update(checkpoint='/not/a/checkpoint',expected_model_hash='wrong')
    with pytest.raises(ValueError,match='model differ'):run(cfg,tmp_path)


def test_stateful_chunked_fixture_replays_and_control_conditions_match(tmp_path):
    from explorer3.sim.branching import experiment
    from explorer3.sim.factory import make
    class QueueFixture:
        metadata={'kind':'test_fixture','learned':False}
        def __init__(self):self.queue=[];self.rng=np.random.default_rng(42);self.calls=0
        def action(self):
            if not self.queue:self.queue=list(self.rng.uniform(-.01,.01,size=(3,2)))
            self.calls+=1;return self.queue.pop(0)
        def snapshot(self):return {'queue':self.queue.copy(),'rng':self.rng.bit_generator.state,'calls':self.calls}
        def restore(self,state):self.queue=state['queue'].copy();self.rng.bit_generator.state=state['rng'];self.calls=state['calls']
        def reset(self):self.queue=[];self.calls=0
    env=make('tank_dock_sim')
    try:
        env.reset(seed=14)
        report=experiment(env,QueueFixture(),tmp_path,config())
        assert report['replay_verified']
        assert all(x is None for x in report['comparisons'].values())
    finally:env.close()
