"""Varied floor objects must collide and rotate as their declared box geometry."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from explorer3.sim.aligned import RosmasterFloorPick


def test_runtime_resizing_matches_independently_compiled_geometry():
    env = RosmasterFloorPick(vary_objects=True)
    try:
        # Reuse one environment through large, flat, narrow, and original objects. This
        # catches stale broad-phase geometry from either the XML or the preceding episode.
        for name in ('bottle', 'cookie', 'block', 'cube'):
            env.reset(seed=9000, options={'object': name})
            half = RosmasterFloorPick.OBJECTS[name]
            reference = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body><freejoint/><geom type="box" size="' +
                ' '.join(map(str, half)) + '" mass=".035"/></body></worldbody></mujoco>')
            geom, body = env.model.geom('cube').id, env.model.body('task_cube').id
            np.testing.assert_allclose(env.model.geom_aabb[geom], reference.geom_aabb[0])
            np.testing.assert_allclose(env.model.body_inertia[body], reference.body_inertia[1])
            np.testing.assert_allclose(env.model.geom_rbound[geom], reference.geom_rbound[0])
            node = env.model.body_bvhadr[body]
            np.testing.assert_allclose(env.model.bvh_aabb[node], reference.bvh_aabb[0])
            # Rebuilding constants must not erase the sampled object's start or the arm pose.
            assert env.data.body('task_cube').xpos[2] == pytest.approx(half[2] + .002)
            np.testing.assert_allclose(env.observation()['state'][:5], env.home_native[:5], atol=3.)
    finally:
        env.close()


def test_new_geometry_is_a_distinct_training_revision():
    import hashlib
    env = RosmasterFloorPick(vary_objects=True)
    try:
        xml, manifest = env.model_builder()
        tree = ET.fromstring(xml)
        custom = tree.find('custom')
        marker = custom.find("numeric[@name='explorer_floor_object_geometry_version']")
        assert marker.get('data') == '2'
        custom.remove(marker)
        assert manifest['xml_sha256'] != hashlib.sha256(ET.tostring(tree, encoding='unicode').encode()).hexdigest()
    finally:
        env.close()
