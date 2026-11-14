"""Observation/action contract shared by simulation, shadow mode and the real robots (M1).

The same function builds the policy input from (a) simulated sensors and (b) the real
sensor feature vector produced by `explorer3.sensors.environment_features`. Any change
of names, order, units or scaling must bump VERSION; policy artifacts store it and the
runtime refuses a mismatch.

Base tasks (Tank, ROSMASTER base) — vector layout, all float32:
  0..2   target: distance/3 m, sin(bearing), cos(bearing)   (last valid detection, robot frame)
  3      target_yaw: marker normal relative to robot heading, / pi
  4      target_visible now (0/1)
  5      target_age: seconds since last valid detection, clipped to 2 s, / 2
  6..41  lidar: 36 sector minimum ranges in the ROBOT frame, clipped [0, 4] m, /4; no return -> 1
  42..44 imu: gyro_z /2 rad/s, accel_x /2 m/s^2, accel_y /2 m/s^2 (gravity-free planar axes)
  45..   previous normalized action (2 for Tank, 3 for ROSMASTER)
Actions are normalized to [-1, 1] and scaled by the robot's POLICY limits below; the robot
agent applies its own clamps, acceleration limit and safety stops on top.
"""
import hashlib
import json

import numpy as np

VERSION = 'base_v1'
LIDAR_BINS = 36
LIDAR_MAX_M = 4.0
TARGET_SCALE_M = 3.0
AGE_CLIP_S = 2.0

ROBOTS = {
    'tank': {'action': ['forward_velocity', 'yaw_velocity'], 'units': ['m/s', 'rad/s'],
             'policy_limits': [0.3, 1.0], 'manual_limits': [0.2, 0.5], 'hard_limits': [0.6, 2.0],
             'accel_limits': [0.5, 2.0], 'control_hz': 20},
    'rosmaster': {'action': ['forward_velocity', 'lateral_velocity', 'yaw_velocity'], 'units': ['m/s', 'm/s', 'rad/s'],
                  'policy_limits': [0.3, 0.25, 0.8], 'manual_limits': [0.3, 0.25, 0.8], 'hard_limits': [0.5, 0.4, 1.5],
                  'accel_limits': [0.5, 0.5, 2.0], 'control_hz': 20},
}

# Indices inside explorer3.sensors.CHANNELS (real feature vector).
REAL_LIDAR = slice(0, 36)
REAL_IMU = slice(36 + 36 + 48 + 48, 36 + 36 + 48 + 48 + 6)  # gyro xyz, accel xyz
REAL_IMU_VALID = 36 + 36 + 48 + 48 + 6


def size(robot):
    return 6 + LIDAR_BINS + 3 + len(ROBOTS[robot]['action'])


def names(robot):
    return (['target_distance', 'target_sin_bearing', 'target_cos_bearing', 'target_yaw', 'target_visible', 'target_age']
            + [f'lidar_{i:02}' for i in range(LIDAR_BINS)] + ['gyro_z', 'accel_x', 'accel_y']
            + [f'previous_{a}' for a in ROBOTS[robot]['action']])


def contract_hash(robot):
    spec = {'version': VERSION, 'robot': robot, 'names': names(robot), 'robot_spec': ROBOTS[robot],
            'scales': [TARGET_SCALE_M, LIDAR_MAX_M, AGE_CLIP_S]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def lidar_to_robot_bins(ranges_lidar_bins, yaw_offset_rad=0.0, direction=1):
    """Re-index 36 lidar-frame sector minima into robot-frame sectors (bin 0 = straight ahead, CCW).

    Real lidars differ in zero angle and rotation direction; both are calibration values in
    configs/scene_measurements.yaml. Shift is rounded to whole 10-degree sectors.
    """
    r = np.asarray(ranges_lidar_bins, dtype=np.float32)
    shift = int(round(yaw_offset_rad / (2 * np.pi / LIDAR_BINS)))
    idx = (direction * np.arange(LIDAR_BINS) + shift) % LIDAR_BINS
    out = np.empty_like(r)
    out[idx] = r
    return out


def normalize_lidar(ranges):
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where((r <= 0) | ~np.isfinite(r), LIDAR_MAX_M, r)  # 0 = no valid return in the real features
    return np.clip(r, 0, LIDAR_MAX_M) / LIDAR_MAX_M


def build(robot, target, lidar_robot_bins, imu, previous_action):
    """target: dict(distance, bearing, yaw, visible, age_s) — last valid detection held by the caller.
    imu: (gyro_z, accel_x, accel_y). previous_action: normalized [-1, 1]."""
    d = float(target['distance'])
    b = float(target['bearing'])
    vec = [min(d, 2 * TARGET_SCALE_M) / TARGET_SCALE_M, np.sin(b), np.cos(b), float(target['yaw']) / np.pi,
           float(target['visible']), min(float(target['age_s']), AGE_CLIP_S) / AGE_CLIP_S]
    out = np.concatenate([np.asarray(vec, np.float32), normalize_lidar(lidar_robot_bins),
                          np.asarray(imu, np.float32) / 2.0, np.asarray(previous_action, np.float32)])
    if out.shape != (size(robot),) or not np.isfinite(out).all():
        raise ValueError('Invalid base observation')
    return out.astype(np.float32)


def from_real_features(robot, features, target, previous_action, lidar_calibration):
    """Real robot: feature vector from explorer3.sensors.environment_features + held marker detection."""
    f = np.asarray(features, dtype=np.float32)
    lidar = lidar_to_robot_bins(f[REAL_LIDAR], lidar_calibration.get('yaw_offset_rad', 0.0),
                                lidar_calibration.get('direction', 1))
    if f[REAL_IMU_VALID] > 0:
        g = f[REAL_IMU]
        imu = (g[2], g[3], g[4])
    else:
        imu = (0.0, 0.0, 0.0)
    return build(robot, target, lidar, imu, previous_action)


def scale_action(robot, normalized):
    a = np.clip(np.asarray(normalized, dtype=np.float32), -1, 1)
    return a * np.asarray(ROBOTS[robot]['policy_limits'], np.float32)
