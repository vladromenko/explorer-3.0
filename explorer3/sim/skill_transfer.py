"""MimicGen-adapted interface for Explorer's existing ROSMASTER environment.

Independent narrow implementation; no upstream Python copied, no alternate simulator.
Teacher-only object poses never enter the policy observation. The donor's native
joint coordinates are neither read nor sent to the recipient.
"""

import mujoco
import numpy as np

from explorer3.skill_packets import admission, retarget_pose, transform
from .aligned import ScriptedFloorPick


class TransferRejected(ValueError):
    """One generated variant is incompatible; not a global feasibility verdict."""


def body_pose(data, name):
    value = np.eye(4)
    body = data.body(name)
    value[:3, :3] = body.xmat.reshape(3, 3)
    value[:3, 3] = body.xpos
    return value


def tcp_pose(data):
    value = np.eye(4)
    site = data.site('gripperframe')
    value[:3, :3] = site.xmat.reshape(3, 3)
    value[:3, 3] = site.xpos
    return value


def so101_jaw_measurement(env):
    # Distance of the paired distal contact pads, not the overlapping proximal
    # jaw bodies. Exact model geometry, explicitly not a physical calibration.
    fixed = env.model.geom('fixed_jaw_box4').id
    moving = env.model.geom('moving_jaw_box2').id
    points = np.zeros(6)
    width = mujoco.mj_geomDistance(env.model, env.data, fixed, moving, .3, points)
    direction = points[3:] - points[:3]
    length = np.linalg.norm(direction)
    if length < 1e-6:
        raise ValueError('Jaw direction cannot be measured at closed contact')
    return max(0., float(width)), direction / length


class RosmasterTransferInterface:
    """The MG_EnvInterface concepts, adapted to native servo targets and limited DOF."""
    def __init__(self, env):
        if not env.profile.startswith('rosmaster_floor_pick'):
            raise ValueError('Only the existing ROSMASTER floor-pick task is supported')
        self.env = env
        self.ik = ScriptedFloorPick(env, knowledge='none')
        self.approach_target = None
        self.relaxation = 'preserve position and horizontal closing axis; use recipient reachable pitch; free symmetric jaw roll'

    def get_robot_eef_pose(self):
        return tcp_pose(self.env.data)

    def get_object_poses(self):
        return {'object': body_pose(self.env.data, 'task_cube')}

    def get_subtask_term_signals(self):
        cube = self.env.data.body('task_cube').xpos
        lift = float(cube[2] - self.env.object_half()[2])
        held = lift > .05 and np.linalg.norm(cube - self.env.data.site('gripperframe').xpos) < .06
        return {'lifted': bool(held), 'recipient_task_complete': bool(self.env.info()['success'])}

    def target_pose_to_action(self, pose, jaw_m, phase, closing_axis, *, initial=False):
        from explorer3.rosmaster_arm import GRIP_OPEN, GRIP_CLOSE, OPEN_WIDTH_M, grip_command
        pose = transform(pose)
        point = pose[:3, 3]
        if point[2] < .012 or point[2] > .5 or np.linalg.norm(point[:2]) > .5:
            raise TransferRejected('Transferred TCP target outside floor clearance/workspace')
        axis = np.asarray(closing_axis, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis[:2]) < .8:
            raise TransferRejected('Donor closing direction incompatible with top grasp')
        self.ik.closing_axis = np.r_[axis[:2] / np.linalg.norm(axis[:2]), 0.]
        # Recipient's own existing metric-width recipe. This is a simulation
        # assumption pending real gripper calibration, reported in every rollout.
        grip = (GRIP_CLOSE - min(OPEN_WIDTH_M, jaw_m) / OPEN_WIDTH_M * (GRIP_CLOSE - GRIP_OPEN)
                if phase == 'approach' else grip_command(jaw_m))
        if initial:
            if self.approach_target is None:
                self.approach_target = self.ik._ik(point, grip, quick=False)
            target = self.approach_target
        else:
            target = self.ik._ik(point, grip, quick=True)
        if np.any(target < self.env.action_space.low) or np.any(target > self.env.action_space.high):
            raise TransferRejected('Recipient joint limit exceeded')
        maximum = np.array([45] * 5 + [120]) * self.env.dt
        self.ik.command += np.clip(target - self.ik.command, -maximum, maximum)
        return self.ik.command.copy()


class RetargetedGrasp:
    """One accepted donor subskill plus the recipient's own approach/carry motions."""
    def __init__(self, env, packet):
        decision = admission(packet)
        if not decision['eligible']:
            raise ValueError('Donor packet incomplete: ' + ', '.join(decision['reasons']))
        if packet['object']['surface'] != 'rigid_box':
            raise ValueError('Only measured rigid box contact geometry is supported in this pilot')
        size = np.asarray(packet['object']['size_m'])
        if np.any(np.abs(2 * env.object_half() / size - 1.) > .2):
            raise ValueError('Recipient object dimensions outside first-transfer scope (20%)')
        self.interface = RosmasterTransferInterface(env)
        self.env, self.packet, self.phase = env, packet, 'own_approach'
        self.initial_target_object = body_pose(env.data, 'task_cube')
        initial_source = transform(packet['phases'][0]['world_object_at_start'])
        self.mapping = self.initial_target_object @ np.linalg.inv(initial_source)
        axis = packet['measurement'].get('closing_axis_object')
        if axis is None:
            raise ValueError('Measured donor closing direction required')
        self.axis = self.initial_target_object[:3, :3] @ np.asarray(axis, dtype=float)
        self.samples = []
        for phase in packet['phases']:
            # Keep the inter-phase offset, including the lift already achieved at
            # the hold boundary. Re-anchoring hold to the floor would lower it again.
            anchor = self.mapping @ transform(phase['world_object_at_start'])
            for sample in phase['samples']:
                self.samples.append({'t': sample['time_s'], 'phase': phase['name'],
                    'pose': retarget_pose(anchor, sample['object_tcp']), 'jaw': sample['jaw_opening_m']})
        start = self.samples[0]['t']
        for sample in self.samples:
            sample['t'] -= start
        self.duration = self.samples[-1]['t']
        self.elapsed = 0.
        self.path_time = 0.
        self.stalled_s = 0.
        self.approach_done = False
        self.trace = []
        self.report = {'method': 'MimicGen-adapted', 'packet_sha256': decision['packet_sha256'],
                       'donor_domain': packet['donor']['domain'], 'donor_behavior_source': packet['donor']['behavior_source'],
                       'source_episodes': packet['donor']['episodes'], 'transform': self.mapping.tolist(),
                       'orientation_relaxation': self.interface.relaxation,
                       'time_adaptation': 'own approach full IK; advance source timeline only within 15 mm TCP tracking tolerance; task deadline unchanged',
                       'rejection_rule': 'reject this variant after 2 s tracking stall; no trajectory correction using moved object',
                       'gripper_mapping': 'existing ROSMASTER metric-width recipe; uncalibrated physical dynamics',
                       'recipient_initial_pose': env.observation()['state'].tolist(),
                       'policy_inputs': 'recipient camera and measured joints only', 'physical_verified': False}

    def action(self):
        t = self.env.elapsed
        initial = not self.approach_done
        if initial:
            sample = self.samples[0]
            phase = 'approach'
            error = np.linalg.norm(self.interface.get_robot_eef_pose()[:3, 3] - sample['pose'][:3, 3])
            self.approach_done = t >= 2.5 and error <= .015
            if t > 6. and not self.approach_done:
                raise TransferRejected('Recipient own approach did not reach the first donor waypoint')
        elif self.path_time <= self.duration:
            sample = min(self.samples, key=lambda sample: abs(sample['t'] - self.path_time))
            phase = sample['phase']
            error = np.linalg.norm(self.interface.get_robot_eef_pose()[:3, 3] - sample['pose'][:3, 3])
            if error <= .015:
                self.path_time += self.env.dt
                self.stalled_s = 0.
            else:
                self.stalled_s += self.env.dt
                if self.stalled_s >= 2.:
                    raise TransferRejected('Transferred path tracking stalled; variant rejected')
        else:
            from explorer3.rosmaster_arm import grip_command
            self.phase = 'own_carry'
            grip = grip_command(self.samples[-1]['jaw'])
            target = np.r_[self.env.cfg['carry_pose'], grip]
            maximum = np.array([45] * 5 + [120]) * self.env.dt
            self.interface.ik.command += np.clip(target - self.interface.ik.command, -maximum, maximum)
            return self.interface.ik.command.copy()
        self.phase = phase
        action = self.interface.target_pose_to_action(sample['pose'], sample['jaw'], phase, self.axis, initial=initial)
        self.trace.append({'time_s': t, 'phase': phase, 'target_tcp_m': sample['pose'][:3, 3].tolist(),
                           'measured_tcp_m': self.interface.get_robot_eef_pose()[:3, 3].tolist(),
                           'measured_object_m': self.env.data.body('task_cube').xpos.tolist(),
                           'source_time_s': self.path_time, 'recipient_action': action.tolist()})
        return action


class TransferVerifier:
    """Measurements after every physics step; independent of the transferred targets."""
    def __init__(self, env):
        self.env = env
        self.floor = env.model.geom('floor').id
        self.max_penetration_m = 0.
        self.forbidden_pairs = set()
        self.held_s = 0.
        self.max_held_s = 0.
        self.lift_m = 0.

    def __call__(self, model, data):
        for contact in data.contact:
            names = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
            arm = lambda name: name.startswith(('arm', 'rlink', 'llink'))
            floor_arm = self.floor in contact.geom and any(arm(name) for name in names)
            arm_self = all(arm(name) for name in names) and not all(name.startswith(('rlink', 'llink')) for name in names)
            if floor_arm or arm_self:
                depth = max(0., -float(contact.dist))
                self.max_penetration_m = max(self.max_penetration_m, depth)
                if depth > .002:
                    self.forbidden_pairs.add(tuple(sorted(names)))
        cube = data.body('task_cube').xpos
        lift = float(cube[2] - self.env.object_half()[2])
        held = lift >= .05 and np.linalg.norm(cube - data.site('gripperframe').xpos) < .06
        self.held_s = self.held_s + model.opt.timestep if held else 0.
        if self.held_s > self.max_held_s:
            self.max_held_s, self.lift_m = self.held_s, lift

    def report(self):
        return {'method': 'every MuJoCo physics substep; independent object lift and TCP proximity',
                'lift_m': self.lift_m, 'held_s': self.max_held_s,
                'max_arm_penetration_m': self.max_penetration_m,
                'forbidden_contact_pairs': [list(pair) for pair in sorted(self.forbidden_pairs)],
                'numerical_contact_tolerance_m': .002,
                'accepted': not self.forbidden_pairs and self.max_held_s >= 1.,
                'not_a_physical_safety_certificate': True}
