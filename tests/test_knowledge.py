import json

import numpy as np

from explorer3 import knowledge
from explorer3.cognitive.gap_advisor import validate
from explorer3.so101_vendor.cuboid_vision import cube_vertices, fit_upright_cube, fit_upright_cuboid, project_vertices


def test_store_assess_and_refute(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, 'STORE', tmp_path / 'k.json')
    it = knowledge.add('cookie', 'so101_station', {'width_m': .07}, success=None)
    assert knowledge.best('cookie')['id'] == it['id']
    knowledge.record_use(it['id'], 'rosmaster', False)
    assert knowledge.best('cookie') is None          # refuted by the recipient's failure
    assert knowledge.helpers_for('how', ['tank', 'so101']) == ['so101', 'human']


def test_advisor_output_is_validated():
    v = validate({'gaps': ['where', 'how', 'fly'], 'helpers': {'how': ['tank', 'so101'], 'where': ['drone']}}, ['tank', 'so101'])
    assert v['gaps'] == ['where', 'how'] and v['rejected'] == ['fly']
    assert v['helpers'] == {'how': ['so101']}         # Tank cannot measure grasps, the drone does not exist
    inv = validate({'gaps': ['how'], 'helpers': {'so101': ['how']}}, ['so101'])
    assert inv['helpers'] == {'how': ['so101']}


def _mask(vertices, calib):
    import cv2
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(project_vertices(vertices, calib).astype(np.int32)), 1)
    return m


def test_copied_fit_measures_a_synthetic_cube_and_box():
    calib = json.load(open('configs/so101_station/overview_calibration.json'))
    _, ev = fit_upright_cube([(_mask(cube_vertices([.25, -.1], .03, .4), calib), calib)])
    assert abs(ev['size_m'] - .03) < .0015
    from explorer3.so101_vendor.cuboid_vision import cuboid_vertices
    box = fit_upright_cuboid(_mask(cuboid_vertices([.22, -.05], (.05, .08, .02), .3), calib), calib, height_m=.02)
    assert abs(box['width_m'] - .05) < .003 and abs(box['length_m'] - .08) < .003
