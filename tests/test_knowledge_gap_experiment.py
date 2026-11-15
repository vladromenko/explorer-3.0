"""Fair solo/team comparisons and durable experiment reports, without any real robot."""
import json
import threading

import numpy as np
import pytest

from explorer3 import fleet_map, sim2d, so101_station, swarm_sim, world_memory
from scripts import knowledge_gap_experiment as experiment


def test_solo_and_team_keep_the_same_initial_scene():
    solo = swarm_sim.SimWorld(7000, active_robots=['rosmaster'])
    team = swarm_sim.SimWorld(7000)
    assert solo.true == team.true
    np.testing.assert_array_equal(solo.pad, team.pad)
    for name in solo.objects:
        np.testing.assert_array_equal(solo.objects[name], team.objects[name])
    assert solo.active_robots == ('rosmaster',)


def test_absent_tank_supplies_no_lidar_camera_or_movement(monkeypatch):
    world = swarm_sim.SimWorld(7000, active_robots=['rosmaster'])
    scans, sightings, moves = [], [], []
    monkeypatch.setattr(fleet_map.FLEET, 'update', lambda robot, *a: scans.append(robot))
    monkeypatch.setattr(sim2d, 'lidar_packet', lambda *a: {})
    monkeypatch.setattr(sim2d, 'visible', lambda *a: (1., 0.))
    monkeypatch.setattr(world_memory, 'record', lambda robot, *a, **kw: sightings.append(robot))
    for pose in world_memory.POSES.values():
        monkeypatch.setattr(pose, 'known', lambda: True)
    monkeypatch.setattr(world, 'move', lambda robot, *a: moves.append(robot))
    monkeypatch.setattr(world.stop_event, 'wait', lambda *a: world.stop_event.set())
    world._run()  # exactly one tick, with both robots present in the paired layout
    assert scans == moves == ['rosmaster']
    assert sightings and set(sightings) == {'rosmaster'}


def test_measurement_noise_is_repeatable_and_does_not_change_navigation(monkeypatch):
    a, b = swarm_sim.SimWorld(7000), swarm_sim.SimWorld(7000)
    monkeypatch.setattr(so101_station.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(so101_station.knowledge, 'add', lambda prompt, source, params, **kw: params)
    first = so101_station.SimSO101Station(a).practice('cookie')['knowledge']
    # Different camera/navigation work must not alter the helper's measurement.
    b.rng.normal(size=1000)
    second = so101_station.SimSO101Station(b).practice('cookie')['knowledge']
    assert first == second
    assert a.grasp_rng.random() == b.grasp_rng.random()


def test_summary_does_not_call_a_completed_pick_a_delivery():
    rows = [dict(seed=1, object='cube', condition='K0', outcome='done', delivered=False, seconds=2.),
            dict(seed=1, object='cube', condition='K3', outcome='done', delivered=True, seconds=4.),
            dict(seed=2, object='sock', condition='K3', outcome='timeout', delivered=False, seconds=600.),
            dict(seed=3, object='ball', condition='K4', outcome='done', delivered=True, seconds=20., advisor_fallback=True)]
    s = experiment.summarize(rows)
    assert s['conditions']['K0']['delivered'] == 0
    assert s['conditions']['K3']['success_rate'] == .5
    assert s['conditions']['K3']['timeouts'] == 1
    assert s['conditions']['K3']['median_seconds_delivered'] == 4.
    assert s['conditions']['K4']['advisor_fallbacks'] == 1
    assert s['paired_vs_K0']['K3'] == {'pairs': 1, 'help_only_successes': 1, 'solo_only_successes': 0, 'both_successes': 0}


def test_resume_skips_finished_episodes_and_refuses_changed_code(tmp_path, monkeypatch):
    out = tmp_path / 'resume'
    out.mkdir()
    protocol = {'version': 2, 'rooms': 1, 'conditions': ['K0'], 'start_seed': 7000,
                'deadline_s': 600., 'objects': list(experiment.OBJECTS), 'sources': experiment.source_hashes()}
    experiment.write_json(out / 'protocol.json', protocol)
    row = dict(seed=7000, object='cookie', condition='K0', outcome='timeout', delivered=False, seconds=600.)
    (out / 'results.jsonl').write_text(json.dumps(row) + '\n')
    monkeypatch.setattr(experiment.sys, 'argv', ['experiment', '--rooms', '1', '--conditions', 'K0', '--resume', str(out)])
    monkeypatch.setattr(experiment, 'isolated_episode', lambda *a: pytest.fail('A finished episode must not run twice'))
    experiment.main()
    monkeypatch.setattr(experiment, 'source_hashes', lambda: {'changed': 'source'})
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2


def test_invalid_participants_do_not_change_live_globals():
    before = swarm_sim.WORLD, dict(swarm_sim._saved), world_memory.MEMORY
    with pytest.raises(ValueError):
        swarm_sim.enter(7000, robots=['typo'])
    assert (swarm_sim.WORLD, swarm_sim._saved, world_memory.MEMORY) == before
