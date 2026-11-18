"""Cartesian Leader -> ROSMASTER mirror: where the SO-101 Leader's gripper is above its table,
ROSMASTER puts its gripper above the floor (same height, radius shifted), solved by ROSMASTER's own IK.

Joint-by-joint 1:1 mirroring cannot reach the floor: SO-101 reaches its table mostly with the elbow,
ROSMASTER (shoulder 0.218 m above the floor) reaches the floor mostly with the shoulder.
Measured anchors (2026-11-18): Leader grasp [-15,-13,51,40,-54] -> tip r 0.243 m, z 0.022 m, pitch 68 deg;
ROSMASTER [90,16,9,85,90] -> TCP x 0.200 (r 0.166 from the arm axis), z 0.015, pitch 70 deg.
Leader straight up -> tip r 0.092, z 0.523; ROSMASTER straight up -> r 0, z 0.544.
"""
import json
import math
from pathlib import Path

import numpy as np

DEFAULT = {'radius_offset_m': -0.077, 'radius_scale': 1.0, 'z_offset_m': -0.007, 'z_scale': 1.0,
           'min_pitch_deg': -90.0, 'max_pitch_deg': 89.0, 'roll_sign': 1, 'roll_leader_ref': 2.2,
           'max_joint_step_deg': 30.0, 'min_interval_s': 0.05, 'pitch_weight': 0.03}


def _rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def _frame(entry):
    t = np.eye(4)
    t[:3, :3] = np.asarray(entry['rotation'], float).reshape(3, 3)
    t[:3, 3] = entry['translation_m']
    return t


class LeaderFK:
    JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')

    def __init__(self, path=Path(__file__).with_name('so101_leader_kinematics.json')):
        data = json.loads(Path(path).read_text())
        self.origins = [_frame(j) for j in data['joints']]
        self.tip = _frame(data['tip'])
        self.offsets = np.asarray(data['offsets_deg'], float)

    def tip_pose(self, degrees):
        t = np.eye(4)
        for origin, angle in zip(self.origins, np.asarray(degrees, float) + self.offsets):
            t = t @ origin @ _rot_z(angle)
        return t @ self.tip


class CartesianMirror:
    def __init__(self, arm_model, arm_cfg, settings=None):
        from cubebot.arm_model import CHAIN
        self.arm, self.a = arm_model, arm_cfg
        self.s = {**DEFAULT, **(settings or {})}
        self.fk = LeaderFK()
        self.axis = np.array(CHAIN[0][0][:2], float)
        self.last = None          # last solved 5 servos
        self.last_at = -1e9
        self.blocked = None

    def target(self, leader):
        """Leader joints (deg) -> (xyz in ROSMASTER base, pitch_down deg, roll servo)."""
        deg = [float(leader[n]) for n in LeaderFK.JOINTS]
        tip = self.fk.tip_pose(deg)
        p, approach = tip[:3, 3], tip[:3, 2]
        s = self.s
        radius = max(0.0, s['radius_offset_m'] + s['radius_scale'] * math.hypot(p[0], p[1]))
        yaw = math.atan2(p[1], p[0])
        z = s['z_offset_m'] + s['z_scale'] * p[2]
        pitch = min(s['max_pitch_deg'], max(s['min_pitch_deg'], math.degrees(math.asin(max(-1., min(1., -approach[2]))))))
        roll = 90.0 + s['roll_sign'] * (deg[4] - s['roll_leader_ref'])
        xy = self.axis + radius * np.array([math.cos(yaw), math.sin(yaw)])
        return np.array([xy[0], xy[1], z]), pitch, min(269., max(1., roll))

    def solve(self, leader, seed, now, full=False):
        """5 ROSMASTER servos for the Leader pose, or the previous solution if unreachable / too soon."""
        if not full and self.last is not None and now - self.last_at < self.s['min_interval_s']:
            return self.last
        xyz, pitch, roll = self.target(leader)
        step = None if full else self.s['max_joint_step_deg']
        # Pitch is a weak preference: position first (with 0.15 the mid-height poses had no solution).
        if self.last is not None:
            seed = [*self.last, 90.0]       # continue from the previous solution, not the lagging servo
        sol, err = self.arm.ik(xyz, pitch, seed, pitch_weight=self.s['pitch_weight'], roll_servo=roll, quick=not full, max_step_deg=step)
        if sol is None and not full:        # fast local step failed (big Leader jump): one global search
            sol, err = self.arm.ik(xyz, pitch, seed, pitch_weight=self.s['pitch_weight'], roll_servo=roll)
        self.last_at = now
        if sol is None:
            self.blocked = f'вне досягаемости ROSMASTER: x {xyz[0]:.2f} z {xyz[2]:.2f} м'
            return self.last
        self.blocked = None
        self.last = [float(v) for v in sol]
        return self.last

    def reset(self):
        self.last, self.last_at = None, -1e9
