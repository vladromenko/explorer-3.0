"""Agent roles inside the swarm: plans of registered skills only, sightings as evidence, help
chosen with recipient-compatible memory, outcomes only from the verifier."""
import pytest

from explorer3 import catalog, fleet_agents
from explorer3.cognitive import store


@pytest.fixture
def agents(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, 'DATA', tmp_path)
    monkeypatch.setattr(store, 'DATA', tmp_path)
    store.initialize()
    a = fleet_agents.FleetAgents(None, domain=lambda: 'sim')
    a.start('fetch', 'cookie', ['rosmaster', 'tank'])
    return a


def _sighting(a, at=1000.):
    return a.sighting({'prompt': 'cookie', 'robot': 'tank', 'room_xy': [1., 2.], 'at': at, 'distance': 1.2, 'uncertainty_m': .2})


def test_plan_uses_registered_skills_and_refuses_impossible(agents):
    skills = [p['skill'] for p in agents.plan]
    assert skills[-5:] == ['rosmaster.fleet_approach.v1', 'rosmaster.fleet_pick_floor.v1', 'rosmaster.fleet_deliver.v1',
                           'rosmaster.fleet_place.v1', 'so101.fleet_handover.v1']
    with pytest.raises(RuntimeError, match='только ROSMASTER'):
        agents.start('fetch', 'cookie', ['tank'])
    with pytest.raises(ValueError, match='Нет такого навыка'):
        agents.card('tank', 'pick_floor')               # Tank has no arm: never invented


def test_same_sighting_is_one_evidence(agents):
    a = _sighting(agents)
    assert a and _sighting(agents) == a
    assert store.evidence([a], fresh=False)[0].payload['object'] == 'cookie'


def test_help_learns_from_verified_cases_and_rejects_incompatible(agents, monkeypatch):
    ev = [_sighting(agents)]
    d = agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)
    assert d['option'] == 'helper_observe' and d['donor'] == 'tank' and 'базовый' in d['why']
    assert agents.verify(d['case'], 'success', 'success', 'success', 'держит') == 'candidate'
    agents.start('fetch', 'cookie', ['rosmaster', 'tank'])
    d2 = agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)
    assert '1 совместимых' in d2['why']
    assert agents.verify(d2['case'], 'success', 'success', 'success', 'держит') == 'candidate'
    # the recipient's skill changed (new code version): old cases are not used for it
    old = agents.card('rosmaster', 'pick_floor')
    agents.skills['rosmaster.fleet_pick_floor.v1'] = old.model_copy(update={'version': 'f' * 64})
    agents.start('fetch', 'cookie', ['rosmaster', 'tank'])
    d3 = agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)
    assert '0 совместимых' in d3['why'] and '2 отклонено как несовместимые' in d3['why']


def test_contradicted_help_is_avoided_and_help_is_limited(agents):
    ev = [_sighting(agents)]
    d = agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)
    assert agents.verify(d['case'], 'failure', 'failure', 'failure', 'Tank не нашёл') == 'contradicted'
    agents.start('fetch', 'cookie', ['rosmaster', 'tank'])
    assert agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)['option'] == 'retry'
    agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)
    assert agents.assist('rosmaster', 'pick_floor', 'cookie', 'pick_failed', ['rosmaster', 'tank'], ev)['option'] == 'human'


def test_roles_report_state(agents):
    names = [r['role'] for r in agents.status()['roles']]
    assert names == fleet_agents.ROLES and agents.status()['plan']
