"""Continuation wiring only; stand-in robots are not physical admission."""
import time
import pytest

from explorer3 import catalog, knowledge, world_memory
from explorer3.cognitive import broker, registry, store
from test_swarm import isolated, make, wait, Arm
from test_mission_goal import zone, measurement


def saved(monkeypatch, domain='sim'):
    request = broker.prepare_acquisition('saved-goal', 'принеси кубик', 'rosmaster', 'cube', {},
        domain=domain, context={'terminal_goal': zone(domain), 'scene': 'old', 'permission': True},
        robots=['rosmaster'])
    request.update(state='accepted', skill_id='fixture-skill')
    with catalog.connect() as db:
        broker._save_acquisition(db, request, 'fixture')
        mission = store.mission('saved-goal')
        mission.update(phase='awaiting_resume')
        db.execute('UPDATE cognitive_missions SET body=? WHERE id=?', (store.encoded(mission), mission['id']))
    monkeypatch.setattr(registry, 'acquired_pick', lambda *a: {
        'state': 'available', 'skill_id': 'fixture-skill', 'job_id': 'fixture-job'})
    return request


def test_resume_is_explicit_single_use_fresh_and_keeps_goal_and_robots(isolated, monkeypatch):
    saved(monkeypatch)
    team, ros, tank = make([])
    team.agents.domain = lambda: 'sim'
    world_memory.POSES['rosmaster'].fix(0., 0., 0.)
    world_memory.record('rosmaster', 'cube', 'cube', 1., 0.)
    observed = []
    def search(prompt, stages):
        observed.append(knowledge.assess('rosmaster', prompt, 'sim', context=team.task_context))
        return {'prompt': prompt, 'robot': 'rosmaster', 'at': time.time(),
                'room_xy': [1., 0.], 'uncertainty_m': .01}
    monkeypatch.setattr(team, '_search', search)
    class MeasuredArm(Arm):
        def observe_delivery(self, prompt):
            return measurement()
    arms = []
    team.arm_factory = lambda log: arms.append(MeasuredArm(log)) or arms[-1]
    team.resume('saved-goal')
    wait(team)
    assert team.state['stage'] == 'done'
    assert observed[0]['readiness']['where_missing']
    assert observed[0]['where'] is None
    assert 'permission' not in team.task_context and 'scene' not in team.task_context
    assert team.task_context['terminal_goal'] == zone()
    assert arms[0].execution_selection == 'fixture-job'
    assert ros.started and not tank.started
    old = store.mission('saved-goal')
    assert old['phase'] == 'resumed' and old['resumed_as'] == team.agents.mission
    assert store.mission(old['resumed_as'])['phase'] == 'completed'
    with pytest.raises(ValueError, match='уже продолжена'):
        team.resume('saved-goal')
    with catalog.connect() as db:
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name='jobs'").fetchone()[0] == 0


def test_real_zone_and_mode_change_refuse_before_motion(isolated, monkeypatch):
    saved(monkeypatch, 'real')
    team, ros, tank = make([])
    with pytest.raises(ValueError, match='реальной доставки'):
        team.resume('saved-goal')
    team.agents.domain = lambda: 'sim'
    with pytest.raises(ValueError, match='другому режиму'):
        team.resume('saved-goal')
    assert not ros.started and not tank.started
    assert store.mission('saved-goal')['phase'] == 'awaiting_resume'


def test_changed_skill_cancel_and_restart_do_not_replay(isolated, monkeypatch):
    saved(monkeypatch)
    monkeypatch.setattr(registry, 'acquired_pick', lambda *a: {'state': 'uncertain'})
    with pytest.raises(ValueError, match='Навык или условия'):
        broker.claim_acquisition_resume('saved-goal', 'sim')
    monkeypatch.setattr(registry, 'acquired_pick', lambda *a: {'state':'available', 'skill_id':'fixture-skill'})
    broker.claim_acquisition_resume('saved-goal', 'sim')
    store.recover_interrupted()
    assert store.mission('saved-goal')['phase'] == 'unknown'
    with pytest.raises(ValueError, match='требует сверки'):
        broker.claim_acquisition_resume('saved-goal', 'sim')
    broker.cancel('saved-goal')
    with pytest.raises(ValueError, match='отменена'):
        broker.claim_acquisition_resume('saved-goal', 'sim')


def test_stop_during_resume_validation_prevents_late_start(isolated, monkeypatch):
    saved(monkeypatch)
    team, ros, tank = make([])
    team.agents.domain = lambda: 'sim'
    def interrupted_validation(*args):
        team.stop('STOP while checking weights')
        return {'state':'available', 'skill_id':'fixture-skill'}
    monkeypatch.setattr(registry, 'acquired_pick', interrupted_validation)
    with pytest.raises(InterruptedError, match='STOP'):
        team.resume('saved-goal')
    assert team.thread is None
    assert not ros.started and not tank.started
    assert store.mission('saved-goal')['phase'] == 'unknown'
