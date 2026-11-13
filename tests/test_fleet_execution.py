"""Actual common MuJoCo world; numerical policies only, no training/hardware."""
import copy
import numpy as np
import pytest
from explorer3.sim.fleet_execution import FleetExecution
from explorer3.sim import state_codec


def test_native_periods_online_monitor_single_slot_and_full_restore(tmp_path):
    runner=FleetExecution({n:{'kind':'fixture'} for n in ('so101','tank')},tmp_path,delay_s=.01)
    try:
        while runner.world.data.time<.19:runner.advance()
        assert sum(r['status']=='assisting' for r in runner.requests.values())<=1
        assert any(e['kind']=='help_started' for e in runner.events)
        assert len([e for e in runner.events if e['kind']=='help_requested'])==2
        state_codec.dump(tmp_path/'saved.json',runner.snapshot());saved=state_codec.load(tmp_path/'saved.json')
        for _ in range(24):runner.advance()
        expected=runner.world.data.qpos.copy();events=copy.deepcopy(runner.events);counts=copy.deepcopy(runner.counts)
        score=copy.deepcopy(runner.last_scores);metrics=runner.metrics()
        runner.restore(saved)
        for _ in range(24):runner.advance()
        np.testing.assert_allclose(runner.world.data.qpos,expected,rtol=0,atol=1e-10)
        assert runner.events==events and runner.counts==counts and runner.last_scores==score
        assert runner.metrics()==metrics and metrics['measured_human_cost'] is False
        assert metrics['waiting_sim_seconds']['tank']>0
        for name,fps in [('so101',30),('tank',20)]:
            assert sum(counts[name].values())==int((runner.world.data.time+1e-8)*fps)
        assert runner.world.render('tank').shape==(400,640,3)
        assert runner.world.render('so101').shape==(480,640,3)
        assert any(e['kind']=='authority_changed' and e['mode']=='policy' for e in events)
        runner.restore(saved,interactive=True)
        with pytest.raises(ValueError,match='paused'):runner.advance()
        runner.resume();runner.advance()
    finally:runner.close()


def test_partial_reset_does_not_change_other_policy_or_physics(tmp_path):
    runner=FleetExecution({n:{'kind':'fixture'} for n in ('so101','tank')},tmp_path)
    try:
        for _ in range(4):runner.advance()
        runner.pause();before=runner.world.data.qpos.copy();policy=runner.policies['so101'].snapshot();clock=runner.world.data.time
        runner.reset_agent('tank',19)
        np.testing.assert_array_equal(runner.world.data.qpos[runner.world.views['so101'].q],before[runner.world.views['so101'].q])
        assert runner.policies['so101'].snapshot()['calls']==policy['calls'] and runner.world.data.time==clock
        assert runner.paused
    finally:runner.close()


def test_correction_capture_uses_pre_step_full_state_and_fixture_cannot_enter_training(tmp_path,monkeypatch):
    from explorer3.sim.fleet_capture import FleetCorrections
    from explorer3.sim import capture
    from explorer3 import fleet_corrections
    monkeypatch.setattr(capture,'DATA',tmp_path);monkeypatch.setattr(fleet_corrections,'DATA',tmp_path)
    runner=FleetExecution({'so101':{'kind':'fixture'},'tank':{'kind':'fixture'}},tmp_path/'run',delay_s=0)
    records=FleetCorrections(runner,1);runner.before_action=records.observe
    try:
        while runner.world.data.time<.45:
            records.prepare_boundary();runner.advance();records.end_completed()
        results=records.finish();runner.before_action=None
        assert results and all(r['frames']>0 for r in results)
        root=__import__('pathlib').Path(results[0]['spool'])
        with pytest.raises(ValueError,match='Программный пример'):fleet_corrections.validate(root)
        saved=state_codec.load(root/'entry-snapshot.json')
        runner.restore(saved);runner.advance()
        expected=runner.world.data.qpos.copy();event=copy.deepcopy(runner.events)
        runner.restore(saved);runner.advance()
        np.testing.assert_array_equal(runner.world.data.qpos,expected)
        assert runner.events==event
    finally:runner.close()


def test_gain_shadow_never_changes_assignment_and_is_part_of_snapshot(tmp_path):
    import json
    from explorer3.research import FEATURES,matrix
    from explorer3.vision_assets import digest
    _,names=matrix([{'features':{n:None for n in FEATURES}}])
    head={'mean':[0.]*len(names),'scale':[1.]*len(names),'coef':[0.]*len(names),'intercept':0.}
    # Explicit numerical inference fixture; no fitting or learned-quality claim.
    artifact={'schema':'branch_gain_model_v1','domain':'sim','supervision':'scripted_expert',
        'protocol':{'correction_s':1.,'horizon_s':1.},'include_physical':True,'feature_names':names,'heads':[head,head]}
    path=tmp_path/'gain.json';path.write_text(json.dumps(artifact));spec={'path':str(path),'sha256':digest(path)}
    runner=FleetExecution({'tank':{'kind':'fixture'}},tmp_path/'run',gain=spec)
    try:
        runner.advance()
        assert 'unavailable' in runner.gain_scores['tank']
        # Exercise inference route without starting a heavyweight policy process.
        runner.policies['tank'].metadata={**runner.policies['tank'].metadata,'kind':'frozen_checkpoint'}
        runner.advance();saved=runner.snapshot();runner.advance();expected=copy.deepcopy(runner.gain_scores)
        assert expected['tank']['controls_assignment'] is False and expected['tank']['human_effect'] is False
        runner.restore(saved);runner.advance();assert runner.gain_scores==expected
    finally:runner.close()
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed after preparation'):
        FleetExecution({'tank':{'kind':'fixture'}},tmp_path/'changed',gain=spec)
