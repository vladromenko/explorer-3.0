"""MuJoCo tasks that mirror the real recording blocks, so that what is rehearsed in simulation is
what will run on the robots (docs/DATASET_RECORDING_RU.md):

  so101_handover_sim     SO-101 from its real rest pose (folded, gripper open) takes the object from
                         the pad (3x3 zones), holds it out at the hand-over point, opens, returns to rest.
                         The person «takes» it: on release near the hand-over point the object is removed.
  rosmaster_floor_pick_sim  ROSMASTER arm from the real observation pose takes the object from the
                         floor 5–10 cm in front of the robot (±5 cm sideways) and lifts it into the
                         carry pose (observation pose with the gripper closed) and holds it.

Joint units, camera and channels are the real profiles' (so101_pick_place / rosmaster_pick_place);
the sim joint zeros/directions were checked against the real calibration (rest pose renders folded).
Scene numbers come from configs/sim-alignment.json — defaults from the instructions, replaced by
scripts/sim_align.py with values measured on real demonstrations.
"""
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from explorer3.paths import ROOT
from .environment import SO101PickPlace
from .models import so101_model, rosmaster_arm_model
from .rosmaster_arm import RosmasterPickPlace

AWAY = '2 2 0'


def alignment(name):
    return json.loads((ROOT / 'configs/sim-alignment.json').read_text())[name]


def _without_box(builder, variant, camera_roll_deg=0.):
    """The task model without the bin (these tasks have none); a new model hash, not the old one.
    camera_roll_deg: the wrist camera turned about its optical axis to the real mount (SO-101:
    the real wrist image shows the jaw entering from the left, upstream Menagerie from below)."""
    def build():
        xml, manifest = builder()
        root = ET.fromstring(xml)
        root.find(".//body[@name='target_box']").set('pos', AWAY)
        if variant == 'rosmaster_floor_pick':
            # v2 rebuilt collision bounds/inertia; v3 varies size within each class,
            # so naming the object alone no longer specifies its dimensions.
            custom = root.find('custom')
            if custom is None:
                custom = ET.SubElement(root, 'custom')
            ET.SubElement(custom, 'numeric', name='explorer_floor_object_geometry_version', data='3')
        for site in root.iter('site'):           # debug markers must not appear in the policy camera
            site.set('rgba', '0 0 0 0')
        if camera_roll_deg:
            from scipy.spatial.transform import Rotation
            cam = root.find(".//camera[@name='wrist_cam']")
            w, x, y, z = [float(v) for v in cam.get('quat').split()]
            q = (Rotation.from_quat([x, y, z, w]) * Rotation.from_euler('z', camera_roll_deg, degrees=True)).as_quat()
            cam.set('quat', f'{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}')
        xml = ET.tostring(root, encoding='unicode')
        return xml, {**manifest, 'xml_sha256': hashlib.sha256(xml.encode()).hexdigest(), 'task_variant': variant,
                     'camera_roll_deg': camera_roll_deg}
    return build


def randomize_looks(env, rng, camera):
    """Wide appearance randomization (Tobin et al. 2017): the real table, light, objects and the
    exact camera are unknown/variable, so the policy must not depend on them. Geometry, physics,
    joint units and signs are never randomized. Returns what was drawn (stored with the episode)."""
    m = env.model
    robot_visual = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_bodyid[g] not in
                    (m.body('task_cube').id, m.body('target_box').id, 0)]
    floor = m.geom('floor').id
    grey = float(rng.uniform(.3, .92))
    tint = rng.uniform(-.04, .04, 3)
    m.geom_matid[floor] = -1
    m.geom_rgba[floor] = [*(np.clip(grey + tint, 0, 1)), 1]
    dark = bool(rng.random() < .75)
    if dark:
        shade = float(rng.uniform(.03, .22))
        for g in robot_visual:
            m.geom_matid[g] = -1
            m.geom_rgba[g] = [shade, shade, shade, 1]
    cube = m.geom('cube').id
    m.geom_rgba[cube] = [*rng.uniform(.05, .95, 3), 1]
    light = float(rng.uniform(.5, 1.4))
    m.light_diffuse[:] *= light
    cid = m.camera(camera).id
    from scipy.spatial.transform import Rotation
    rot = rng.uniform(-5, 5, 3)
    m.cam_quat[cid] = (Rotation.from_quat(m.cam_quat[cid][[1, 2, 3, 0]]) * Rotation.from_euler('xyz', rot, degrees=True)).as_quat()[[3, 0, 1, 2]]
    m.cam_pos[cid] += rng.uniform(-.005, .005, 3)
    fov = float(rng.uniform(55, 80)) if camera == 'wrist_cam' else float(m.cam_fovy[cid] + rng.uniform(-3, 3))
    m.cam_fovy[cid] = fov
    return {'floor_grey': grey, 'dark_robot': dark, 'light': light, 'camera_rotation_deg': rot.tolist(), 'camera_fovy': fov}


class SO101Handover(SO101PickPlace):
    profile = 'so101_handover_sim'
    model_builder = staticmethod(_without_box(so101_model, 'so101_handover', camera_roll_deg=90.))

    def __init__(self, **kwargs):
        self.cfg = alignment('so101_handover')
        self.home_native = np.array(self.cfg['rest_pose'], float)
        kwargs.setdefault('deadline_s', 30.)
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        c, rng = self.cfg, self.np_random
        rest = self.home_native + np.r_[rng.uniform(-1, 1, 5) * c['rest_noise_deg'], 0.]
        lo, hi = self.from_radians(self.ranges[:, 0]), self.from_radians(self.ranges[:, 1])
        rest = np.clip(rest, np.minimum(lo, hi), np.maximum(lo, hi))
        self.data.qpos[:6] = self.to_radians(rest)
        free = int(self.model.joint('cube_free').qposadr[0])
        if free == 7:
            self.data.qpos[6] = self.data.qpos[5]
        zone = rng.integers(-1, 2, 2) * c['pad_zone_step_m']
        xy = np.array(c['pad_center']) + zone + rng.uniform(-1, 1, 2) * c['pad_jitter_m']
        self.data.qpos[free:free + 3] = [xy[0], xy[1], .018]
        self.data.qpos[free + 3:free + 7] = [1, 0, 0, 0]
        self.data.ctrl[:] = self.data.qpos[:6]
        self.filtered, self.pending = self.data.ctrl.copy(), self.data.ctrl.copy()
        self.handed = False
        self.lifted = False
        self.rest = rest
        self.episode_parameters['looks'] = randomize_looks(self, rng, self.camera_name)
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), self.info()

    def handover_point(self):
        return np.array(self.cfg['handover_point'], float)

    def info(self):
        base = super().info()
        state = self.from_radians(self.data.qpos[:6])
        returned = bool(np.abs(state[:4] - self.rest[:4]).max() < self.cfg['return_tolerance_deg']) if hasattr(self, 'rest') else False
        base.update(handed=bool(getattr(self, 'handed', False)), returned=returned, inside_box=False,
                    success=bool(getattr(self, 'handed', False)) and self.stable >= .5, task='so101_handover')
        return base

    def complete_step(self, target):
        self.pending = target
        self.steps += 1
        self.elapsed = self.data.time
        cube = self.data.body('task_cube').xpos.copy()
        ee = self.data.site('gripperframe').xpos.copy()
        holding = np.linalg.norm(ee - cube) < .05
        if cube[2] > .06 and holding and not self.lifted:
            self.lifted = True
            self.grip_hold = float(self.data.qpos[5])               # fingers stopped by the object
        opening = self.lifted and float(self.pending[5]) > self.grip_hold + .08     # the hand is opening
        if self.lifted and not self.handed and opening and np.linalg.norm(cube - self.handover_point()) < self.cfg['handover_tolerance_m']:
            # the person takes it: the object leaves the scene
            self.handed = True
            free = int(self.model.joint('cube_free').qposadr[0])
            self.data.qpos[free:free + 3] = [1.5, 1.5, .018]
            self.data.qvel[int(self.model.joint('cube_free').dofadr[0]):][:6] = 0
            mujoco.mj_forward(self.model, self.data)
        info = self.info()
        self.stable = self.stable + self.dt if self.handed and info['returned'] else 0.
        info = self.info()
        self.phase = 'return' if self.handed else 'deliver' if self.lifted else 'approach'
        dropped = not self.handed and self.lifted and cube[2] < .03 and not holding
        deadline = self.steps * self.dt + 1e-9 >= self.deadline_s
        success = info['success']
        terminated = success or dropped
        truncated = deadline and not terminated
        info['outcome'] = 'success' if success else 'failure' if terminated or truncated else None
        info['termination_reason'] = ('handed_over_and_returned' if success else 'dropped_before_handover' if dropped
                                      else 'task_deadline' if truncated else None)
        reward = -float(np.linalg.norm(ee - cube)) * (not self.lifted) - float(np.linalg.norm(ee - self.handover_point())) * self.lifted
        return self.observation(), reward + 10. * success, terminated, truncated, info


class RosmasterFloorPick(RosmasterPickPlace):
    profile = 'rosmaster_floor_pick_sim'
    model_builder = staticmethod(_without_box(rosmaster_arm_model, 'rosmaster_floor_pick'))

    # Knowledge-gap root in MuJoCo: object half-extents (x, y, z) in metres. Modelled household objects
    # (assumed sizes, not measurements of the user's objects); the 32 mm cube is the original task.
    OBJECTS = {'cube': (.016, .016, .016), 'cookie': (.035, .035, .006), 'sock': (.04, .025, .012),
               'bottle': (.02, .02, .045), 'block': (.03, .015, .015), 'ball': (.022, .022, .022)}
    GRASPABLE = ('cube', 'bottle', 'block', 'ball')
    LOOK_POSE = (90., 130., 0., -15., 90., 30.)

    def __init__(self, vary_objects=False, with_knowledge=False, **kwargs):
        self.with_knowledge = with_knowledge
        if vary_objects:
            self.profile = 'rosmaster_floor_pick_' + ('knowledge' if with_knowledge else 'noknowledge') + '_sim'
        self.cfg = alignment('rosmaster_floor_pick')
        self.home_native = np.array(self.cfg['start_pose'], float)
        if vary_objects:
            # The aligned start pose (wrist pitch 25) looks over the grasp window: the object was in the
            # arm camera in 0/30 scenes, so a policy could not aim (diagnosis 16.11: 0/20, jaws closed
            # 5 cm from the cube). Wrist pitch -15 sees it in 30/30 scenes at the same shoulder.
            self.home_native = np.array(self.LOOK_POSE, float)
        self.vary_objects = vary_objects
        kwargs.setdefault('deadline_s', 25.)
        super().__init__(**kwargs)

    def object_half(self):
        return self.model.geom_size[self.model.geom('cube').id].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        c, rng = self.cfg, self.np_random
        start = self.home_native + np.r_[rng.uniform(-1, 1, 5) * c['start_noise_deg'], 0.]
        self.data.qpos[:6] = self.to_radians(start)
        self.data.qpos[6:11] = self.data.qpos[5] * np.array([1, 1, -1, -1, 1])
        free = int(self.model.joint('cube_free').qposadr[0])
        g = self.model.geom('cube').id
        name = 'cube'
        if self.vary_objects:
            # demonstrations/training sample the objects this gripper can take from above; flat ones
            # (cookie 1.2 cm, sock) stay available by name for the experiments (options={'object': ...})
            name = (options or {}).get('object') or str(rng.choice(self.GRASPABLE))
        half = np.array(self.OBJECTS[name])
        scale = np.ones(3)
        if self.vary_objects:
            scale = np.asarray((options or {}).get('size_scale', rng.uniform(.8, 1.2, 3)), dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any((scale < .6) | (scale > 1.4)):
                raise ValueError('Object size scale must have three finite values in [.6, 1.4]')
            half *= scale
        self.model.geom_size[g] = half
        # Changing geom_size alone leaves MuJoCo's broad-phase boxes/BVH at the old cube
        # dimensions. Rebuild those constants and the uniform box's inertia. The mass is
        # still the explicit model assumption (35 g, with the selected variation), not measured.
        body = self.model.body('task_cube').id
        self.model.body_inertia[body] = self.model.body_mass[body] / 3 * np.array(
            [half[1] ** 2 + half[2] ** 2, half[0] ** 2 + half[2] ** 2, half[0] ** 2 + half[1] ** 2])
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.object_name = name
        self.episode_parameters['object'] = {'name': name, 'size_m': (2 * half).round(4).tolist(),
                                             'geometry': 'box_proxy', 'mass_kg': float(self.model.body_mass[body]),
                                             'geometry_version': 3, 'size_scale': scale.tolist()}
        # task text: with the helper's knowledge (measured size, ±1.5 mm like the SO-101 station) or without
        self.task_text = None
        if self.vary_objects:
            size = 2 * half + rng.normal(0, .0015, 3)
            self.task_text = (f'pick up the {name}, {size[0] * 100:.1f} x {size[1] * 100:.1f} cm, {size[2] * 100:.1f} cm high'
                              if self.with_knowledge else f'pick up the {name}')
        self.data.qpos[free:free + 3] = [rng.uniform(*c['object_x_m']), rng.uniform(*c['object_y_m']), half[2] + .002]
        yaw = rng.uniform(-np.pi, np.pi) if self.vary_objects else 0.
        self.data.qpos[free + 3:free + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.ctrl[:] = self.data.qpos[:6]
        self.filtered, self.pending = self.data.ctrl.copy(), self.data.ctrl.copy()
        self.episode_parameters['looks'] = randomize_looks(self, rng, self.camera_name)
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), self.info()

    def info(self):
        base = super().info()
        cube = self.data.body('task_cube').xpos
        ee = self.data.site('gripperframe').xpos
        state = self.from_radians(self.data.qpos[:6])
        if self.vary_objects:     # wider objects stop the jaws earlier: held = lifted off the floor at the gripper
            held = bool(np.linalg.norm(ee - cube) < .06 and cube[2] - self.object_half()[2] > self.cfg['lift_min_m'] - .016)
        else:
            held = bool(not self.gripper_is_open() and np.linalg.norm(ee - cube) < .05 and cube[2] > self.cfg['lift_min_m'])
        carry = bool(np.abs(state[:5] - np.array(self.cfg['carry_pose'])).max() < self.cfg['carry_tolerance_deg'])
        base.update(held=held, in_carry_pose=carry, inside_box=False, task='rosmaster_floor_pick',
                    success=self.stable >= self.cfg['hold_s'])
        return base

    def complete_step(self, target):
        self.pending = target
        self.steps += 1
        self.elapsed = self.data.time
        info = self.info()
        self.stable = self.stable + self.dt if info['held'] and info['in_carry_pose'] else 0.
        info = self.info()
        cube = self.data.body('task_cube').xpos
        ee = self.data.site('gripperframe').xpos
        self.phase = 'carry' if info['held'] else 'approach'
        success = info['success']
        deadline = self.steps * self.dt + 1e-9 >= self.deadline_s
        fallen = np.linalg.norm(cube[:2]) > 1
        terminated = success or fallen
        truncated = deadline and not terminated
        info['outcome'] = 'success' if success else 'failure' if terminated or truncated else None
        info['termination_reason'] = 'held_in_carry_pose' if success else 'task_deadline' if truncated else 'object_lost' if fallen else None
        reward = -float(np.linalg.norm(ee - cube)) + float(info['held'])
        return self.observation(), reward + 10. * success, terminated, truncated, info


class ScriptedHandover:
    """Privileged geometric teacher for so101_handover_sim (not a learned policy, not a human)."""
    def __init__(self, env):
        from .expert import ScriptedSO101
        self.env = env
        self.ik = ScriptedSO101(env)              # its IK (keeps the start orientation of the gripper)
        self.cube = env.data.body('task_cube').xpos.copy()
        self.point = env.handover_point()
        self.rest = env.rest.copy()
        self.command = env.observation()['state'].astype(float).copy()
        self.targets = {}
        self.phase = 'approach'
        # the gripper orientation the original pick expert grasps with (its raised start pose)
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:6] = env.to_radians(SO101PickPlace.home_native)
        mujoco.mj_kinematics(env.model, scratch)
        self.ik.orientation = scratch.site('gripperframe').xmat.reshape(3, 3).copy()

    def position_ik(self, xyz, gripper):
        """Hand-over point: position only (how the object is held out to a person does not matter)."""
        m, d = self.env.model, self.ik.scratch
        d.qpos[:] = self.env.data.qpos
        jacp = np.zeros((3, m.nv))
        for _ in range(150):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = xyz - d.site('gripperframe').xpos
            if np.linalg.norm(err) < .002:
                break
            mujoco.mj_jacSite(m, d, jacp, None, m.site('gripperframe').id)
            jac = jacp[:, :4]                       # wrist roll does not move the point
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(3) * 1e-4, err)
            d.qpos[:4] = np.clip(d.qpos[:4] + np.clip(delta, -.1, .1), self.env.ranges[:4, 0], self.env.ranges[:4, 1])
        q = d.qpos[:6].copy()
        q[5] = gripper
        return self.env.from_radians(q)

    def action(self):
        t = self.env.elapsed
        open_, closed = .9, -.12
        # out of the folded rest pose first (straight to the object would sweep it off the pad)
        plan = [(2., 'raise', self.cube * [.6, .6, 0] + [0, 0, .16], open_), (4., 'approach', self.cube + [0, 0, .09], open_),
                (6., 'descend', self.cube + [0, 0, .002], open_), (7.5, 'grasp', self.cube + [0, 0, .002], closed),
                (9., 'lift', self.cube + [0, 0, .12], closed), (12., 'reach', self.point, closed),
                (13.5, 'hold', self.point, closed), (14.5, 'release', self.point, open_)]
        phase, target, grip = 'return', None, open_
        for end, name, xyz, g in plan:
            if t < end:
                phase, target, grip = name, xyz, g
                break
        if phase not in self.targets:
            if target is None:
                self.targets[phase] = self.rest.copy()
            elif phase in {'reach', 'hold', 'release'}:
                self.targets[phase] = self.position_ik(np.asarray(target, float), grip)
            else:
                self.targets[phase] = self.ik.solve(np.asarray(target, float), grip)
        self.phase = phase
        speed = 15 if phase == 'reach' else 25 if phase == 'return' else 50
        maximum = np.array([speed] * 5 + [90]) * self.env.dt
        self.command += np.clip(self.targets[phase] - self.command, -maximum, maximum)
        return self.command.copy()


class ScriptedFloorPick:
    """Privileged teacher for rosmaster_floor_pick_sim: the console IK, as in the real arm skill.

    knowledge: None -> privileged (true object centre, the original teacher);
               'none' -> the recipient knows WHERE (x, y) but not the object: assumes the 32 mm cube
                         (grasp height 16 mm, default jaw opening/closing), as ROSMASTER without help;
               dict  -> grasp knowledge from a helper ({'width_m', 'height_m', ...}, e.g. the SO-101
                         station): grasp height = half the height, jaw closing for that width.
    """
    def __init__(self, env, knowledge=None):
        from .vendor_rosmaster.arm_model import ArmModel
        self.env = env
        self.arm = ArmModel(tcp_offset=float(env.model.site('gripperframe').pos[2]))
        self.cube = env.data.body('task_cube').xpos.copy()
        self.carry = list(env.cfg['carry_pose'])
        self.open, self.close = 30, 150
        self.closing_axis = None
        if env.vary_objects and knowledge is None:
            # Only the full-information teacher knows the object's narrow axis. A helper
            # carrying size alone must not silently receive the true orientation as well.
            rotation = env.data.body('task_cube').xmat.reshape(3, 3)
            self.closing_axis = rotation[:, int(np.argmin(env.object_half()[:2]))].copy()
        if knowledge is not None:
            from ..rosmaster_arm import grip_command, GRIP_OPEN
            if knowledge == 'none':
                height, width = .032, .032
            else:
                height, width = float(knowledge['height_m']), float(knowledge['width_m'])
                if knowledge.get('closing_yaw_deg') is not None:
                    # E2: direction to close across, as the recipient's own camera estimates it (with error)
                    a = np.radians(float(knowledge['closing_yaw_deg']))
                    self.closing_axis = np.array([np.cos(a), np.sin(a), 0.])
            self.cube = np.r_[self.cube[:2], height / 2]
            self.open, self.close = GRIP_OPEN, grip_command(width)
        self.command = env.observation()['state'].astype(float).copy()
        self.targets = {}
        self.phase = 'approach'

    def _ik(self, xyz, grip, quick=False):
        solution = None
        if quick:
            solution, _ = self.arm.ik(xyz, 70, self.command, quick=True)
        if solution is None:
            solution, _ = self.arm.grasp_ik(xyz, self.env.observation()['state'])
        if solution is None:
            raise RuntimeError('ROSMASTER IK cannot reach the floor object')
        if self.closing_axis is not None:
            rotation = self.arm.fk(solution)['arm5'][:3, :3]
            angle = np.degrees(np.arctan2(-np.dot(self.closing_axis, rotation[:, 0]),
                                          np.dot(self.closing_axis, rotation[:, 1])))
            # Symmetric jaws: modulo 180 keeps the wrist inside its 0..270 degree range.
            solution[4] = (angle + 90) % 180
        return np.r_[solution, grip]

    def action(self):
        t = self.env.elapsed
        o, c = self.open, self.close
        if self.env.vary_objects and (2.5 <= t < 4.5 or 6.5 <= t < 8.):
            # A joint-space shortcut from hover to grasp sweeps the open fingers through
            # wide/tall objects and pushes them away before closing. Follow a vertical TCP
            # path instead, retaining the same speed limits and phase deadlines.
            if t < 4.5:
                phase, fraction, z0, z1, grip = 'descend', (t - 2.5) / 1.7, .10, .004, o
            else:
                phase, fraction, z0, z1, grip = 'lift', (t - 6.5) / 1.2, .004, .12, c
            target = self._ik(self.cube + [0, 0, z0 + (z1 - z0) * min(1., fraction)], grip, quick=True)
            self.phase = phase
            maximum = np.array([45] * 5 + [120]) * self.env.dt
            self.command += np.clip(target - self.command, -maximum, maximum)
            return self.command.copy()
        plan = [(2.5, 'approach', self.cube + [0, 0, .10], o), (4.5, 'descend', self.cube + [0, 0, .004], o),
                (6.5, 'grasp', self.cube + [0, 0, .004], c), (8., 'lift', self.cube + [0, 0, .12], c)]
        phase, target, grip = 'carry', None, c
        for end, name, xyz, g in plan:
            if t < end:
                phase, target, grip = name, xyz, g
                break
        if phase not in self.targets:
            self.targets[phase] = (np.array(self.carry + [grip], float) if target is None
                                   else self._ik(np.asarray(target, float), grip))
        self.phase = phase
        maximum = np.array([45] * 5 + [120]) * self.env.dt
        self.command += np.clip(self.targets[phase] - self.command, -maximum, maximum)
        return self.command.copy()
