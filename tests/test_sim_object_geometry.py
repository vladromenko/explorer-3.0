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


def test_knowledge_and_control_receive_the_same_scene_but_different_task_text():
    from explorer3.sim.factory import make
    known = make('rosmaster_floor_pick_knowledge_sim')
    control = make('rosmaster_floor_pick_noknowledge_sim')
    try:
        for seed in (7000, 7001):
            a, _ = known.reset(seed=seed)
            b, _ = control.reset(seed=seed)
            np.testing.assert_array_equal(known.data.qpos, control.data.qpos)
            np.testing.assert_array_equal(known.object_half(), control.object_half())
            assert known.episode_parameters == control.episode_parameters
            assert known.task_text.startswith(control.task_text + ', ')
            assert 'cm high' in known.task_text and 'cm' not in control.task_text
            assert set(a) == set(b) == {'state'}  # no privileged object data in the observation
    finally:
        known.close()
        control.close()


@pytest.mark.parametrize('name', RosmasterFloorPick.GRASPABLE)
def test_full_information_teacher_lifts_wide_objects_without_sweeping_them_away(name):
    from explorer3.sim.aligned import ScriptedFloorPick
    env = RosmasterFloorPick(vary_objects=True)
    try:
        env.reset(seed=9000, options={'object': name})
        expert = ScriptedFloorPick(env)
        # Size-only helpers and no-help control do not get a hidden orientation oracle.
        assert ScriptedFloorPick(env, knowledge='none').closing_axis is None
        assert ScriptedFloorPick(env, knowledge={'width_m': .04, 'height_m': .04}).closing_axis is None
        previous = expert.command.copy()
        for _ in range(round(env.fps * env.deadline_s) + 1):
            action = expert.action()
            assert np.all(np.abs(action - previous) <= np.array([45] * 5 + [120]) * env.dt + 1e-8)
            previous = action.copy()
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert info['success'], info
        assert env.data.body('task_cube').xpos[2] > .1
    finally:
        env.close()
