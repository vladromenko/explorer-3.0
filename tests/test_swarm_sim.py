"""«Рой без роботов»: the virtual room runs the real swarm code and leaves real data untouched."""
import json
import time

import numpy as np
import pytest

from explorer3 import episodes, fleet_map as fm, sim2d, so101_handoff, swarm, swarm_pages, swarm_sim, world_memory


class _NoLocalizer:
    threads = []
    done = None

    def running(self):
        return False

    def stop(self):
        pass

    def start(self):
        raise AssertionError('was not running: must not be started')


@pytest.fixture
def real(monkeypatch, tmp_path):
    monkeypatch.setattr(fm, 'MAP_DIR', tmp_path / 'map')
    monkeypatch.setattr(fm, 'LOCALIZER', _NoLocalizer())
    monkeypatch.setattr(world_memory, 'MEMORY', tmp_path / 'memory.json')
    monkeypatch.setattr(episodes, 'ROOT', tmp_path / 'episodes')
    monkeypatch.setattr(so101_handoff, 'CONFIG', tmp_path / 'handoff.json')
    monkeypatch.setattr(swarm_sim, 'SIM', tmp_path / 'sim')
    from explorer3 import catalog
    from explorer3.cognitive import store
    monkeypatch.setattr(catalog, 'DATA', tmp_path)
    monkeypatch.setattr(store, 'DATA', tmp_path)
    store.initialize()
    monkeypatch.setattr(swarm, 'SWARM', swarm.Swarm())
    # «real» data that must survive the simulation
    fm.FLEET.reset()
    fm.FLEET.grid.hits[200, 210] = 3.
    fm.FLEET.grid.scans = 7
    fm.FLEET.save()
    so101_handoff.save({'rosmaster_spot': [1., 2., 3.]})
    world_memory.POSES['rosmaster'].fix(0., 0., 0.)
    world_memory.record('rosmaster', 'настоящий', 'sock', 1., 0.)
    yield tmp_path
    if swarm_sim.WORLD is not None:
        swarm_sim.leave()
    fm.FLEET.reset()


def test_free_spots_are_never_inside_furniture():
    rng = np.random.default_rng(8)
    room = sim2d.make_room(rng)
    for _ in range(300):
        assert not sim2d.inside_furniture(room, sim2d.free_spot(room, rng, clearance=.3))


def test_enter_and_leave_keep_real_data(real):
    real_swarm = swarm.SWARM
    team = swarm_sim.enter(seed=11)
    try:
        assert swarm.SWARM is team and swarm_sim.WORLD is not None
        assert world_memory.MEMORY.parent == swarm_sim.SIM and episodes.ROOT.parent == swarm_sim.SIM
        assert so101_handoff.load().get('simulation') and not so101_handoff.missing()
        assert swarm_pages.readiness(None)[0][0] == 'Симуляция'
        time.sleep(1.)
        assert fm.FLEET.grid.scans > 0 and fm.FLEET.grid.hits[200, 210] == 0     # a virtual map, not the real one
    finally:
        swarm_sim.leave()
    assert swarm.SWARM is real_swarm and swarm_sim.WORLD is None
    assert so101_handoff.load() == {'rosmaster_spot': [1., 2., 3.]}
    assert [i['text'] for i in world_memory.known_objects()] == ['настоящий']
    assert fm.FLEET.grid.scans == 7 and fm.FLEET.grid.hits[200, 210] == 3.
    assert not (real / 'episodes').exists()


def test_team_finds_an_object_in_the_virtual_room(real):
    team = swarm_sim.enter(seed=11)
    team.command('найдите носок')
    end = time.monotonic() + 240
    while team.thread.is_alive() and time.monotonic() < end:
        time.sleep(.5)
    st = team.status()
    assert st['stage'] == 'done', st['message']
    assert swarm_sim.truth_overlay()['collisions'] == 0
    index = (swarm_sim.SIM / 'episodes' / 'index.jsonl').read_text().splitlines()
    assert json.loads(index[-1])['kind'] == 'team_find'


def test_search_trial_is_journaled_and_confirmed_by_the_operator(real, monkeypatch):
    from explorer3 import research_log
    monkeypatch.setattr(research_log, 'LOG', real / 'research' / 'search_trials.jsonl')
    team = swarm_sim.enter(seed=11)
    team.search_trial('sock', 'A3', 'S01', 200)
    end = time.monotonic() + 240
    while team.thread.is_alive() and time.monotonic() < end:
        time.sleep(.5)
    rows = research_log.rows()
    assert len(rows) == 1 and rows[0]['condition'] == 'A3' and rows[0]['world'] == 'sim' and rows[0]['confirmed'] is None
    assert rows[0]['seconds'] <= 200
    research_log.confirm(rows[0]['trial_id'], 'верно', 0)
    assert research_log.rows()[0]['confirmed'] == 'верно' and 'A3' in research_log.to_csv()
