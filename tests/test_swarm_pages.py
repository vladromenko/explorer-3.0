"""The swarm screen's map: robots, objects, the SO-101 station and the camera-seen floor."""
import io
import math
import time

import pytest

PIL = pytest.importorskip('PIL')
from explorer3 import episodes, fleet_map as fm, so101_handoff, swarm_pages, world_memory


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    f = fm.FleetMap()
    monkeypatch.setattr(fm, 'FLEET', f)
    monkeypatch.setattr(world_memory, 'MEMORY', tmp_path / 'memory.json')
    monkeypatch.setattr(episodes, 'ROOT', tmp_path / 'episodes')
    monkeypatch.setattr(so101_handoff, 'CONFIG', tmp_path / 'handoff.json')
    return f


def test_empty_map_renders_nothing():
    assert swarm_pages.render_map() is None


def test_map_has_robots_objects_station_and_seen_floor(isolated):
    import sys
    sys.path.insert(0, 'tests')
    from test_navigation import packet
    isolated.update('rosmaster', packet('rosmaster', (0., 0., 0.)), (0., 0., 0.))
    isolated.track['rosmaster']['path'] = [(.5, .3), (1., .5)]
    assert isolated.grid.seen.any()                                   # camera cone marked while tracking
    world_memory.POSES['rosmaster'].fix(0., 0., 0.)
    world_memory.record('rosmaster', 'к печенью', 'cookie', 1.0, .2)
    so101_handoff.teach('rosmaster_spot', [-1.0, -.5, math.pi])
    png = swarm_pages.render_map()
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    assert im.size[0] > 200 and im.size[1] > 200
    px = list(im.convert('RGB').getdata())
    assert (30, 62, 66) in px                                         # seen floor
    assert any(p == swarm_pages.COLORS['so101'] for p in px)          # station
    assert any(p == swarm_pages.COLORS['rosmaster'] for p in px)
