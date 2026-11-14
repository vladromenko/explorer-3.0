"""Batched GPU pick-and-place environment for the arms on MuJoCo Warp (M5, M6).

Physics is the same MuJoCo model as explorer3.sim.environment (Menagerie SO-101 +
cube + open box, elliptic contacts), stepped on the GPU for N worlds at once with the
control period captured in a CUDA graph. The command filter that the CPU environment
applies in Python every substep is expressed as the actuators' own first-order
dynamics (timeconst), so the whole control period runs inside MuJoCo.

Observation contract `arm_v1` (actor; identical on the real arm):
  0..5   joint positions, normalized to [-1, 1] over the actuator range
  6..11  joint velocities / 3 rad/s
  12..17 current joint command (normalized like positions)
  18..20 cube position in the arm base frame / 0.3 m   (from perception on the real robot)
  21     cube visible now, 22 cube detection age / 1 s
  23..25 box position in the base frame / 0.3 m          (scene constant)
  26..31 previous normalized action
Action: normalized delta of the joint command, scaled by MAX_STEP per control period.
"""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

CONTROL_HZ = 30
SUBSTEPS = 67
MAX_STEP = np.deg2rad([2.0, 2.0, 2.0, 2.5, 3.0, 6.0])   # per 1/30 s
ARM_OBS = 32
ROBOTS = {
    'so101': {'hz': 30, 'substeps': 67, 'max_step_deg_s': [60, 60, 60, 75, 90, 180]},
    'rosmaster': {'hz': 20, 'substeps': 100, 'max_step_deg_s': [60, 60, 60, 75, 90, 180]},
}


def build_model(robot='so101', lag_s=0.04):
    from explorer3.sim import models
    xml, manifest = (models.so101_model if robot == 'so101' else models.rosmaster_arm_model)()
    root = ET.fromstring(xml)
    for act in root.find('actuator'):
        if act.tag in ('position', 'general'):
            act.set('timeconst', str(lag_s))   # first-order command filter inside MuJoCo
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    spec = ROBOTS[robot]
    model.opt.timestep = 1.0 / (spec['hz'] * spec['substeps'])
    return model, manifest


class ArmBatch:
    home_deg = np.array([-11.00079, -15.01149, 14.49583, 64.74423, 4.54356, 50.76047])

    def __init__(self, n=1024, seed=0, curriculum=0.0, deadline_s=20.0, nconmax=None, njmax=None, robot='so101'):
        import warp as wp
        import mujoco_warp as mjw
        self.wp, self.mjw = wp, mjw
        self.n, self.curriculum = n, curriculum
        self.device = torch.device('cuda')
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        self.robot = robot
        spec = ROBOTS[robot]
        self.substeps = spec['substeps']
        self.mjm, self.manifest = build_model(robot)
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.put_data(self.mjm, mjd, nworld=n, nconmax=nconmax, njmax=njmax)
        T = wp.to_torch
        self.qpos, self.qvel = T(self.d.qpos), T(self.d.qvel)
        self.ctrl, self.act = T(self.d.ctrl), T(self.d.act)
        self.xpos, self.site_xpos = T(self.d.xpos), T(self.d.site_xpos)
        self.cube_body = self.mjm.body('task_cube').id
        self.box_body = self.mjm.body('target_box').id
        self.grip_site = self.mjm.site('gripperframe').id
        self.cube_q = int(self.mjm.joint('cube_free').qposadr[0])
        self.cube_v = int(self.mjm.joint('cube_free').dofadr[0])
        rng = self.mjm.actuator_ctrlrange
        self.lo = torch.tensor(rng[:, 0], dtype=torch.float32, device=self.device)
        self.hi = torch.tensor(rng[:, 1], dtype=torch.float32, device=self.device)
        self.max_step = torch.tensor(np.deg2rad(spec['max_step_deg_s']) / spec['hz'], dtype=torch.float32, device=self.device)
        if robot == 'so101':
            home = np.deg2rad(self.home_deg)
            home[5] = rng[5, 0] + self.home_deg[5] / 100 * np.ptp(rng[5])
            self.open_threshold = 0.45
        else:
            from explorer3.sim.rosmaster_arm import RosmasterPickPlace
            closed = self.manifest['gripper_closed_rad']
            servo = RosmasterPickPlace.home_native
            home = np.deg2rad(servo - 90)
            home[0] = np.deg2rad(90 - servo[0])
            home[5] = (servo[5] - 30) / 95 * closed
            self.open_threshold = 0.5 * closed
        self.home = torch.tensor(home, dtype=torch.float32, device=self.device)
        self.mimic = robot == 'rosmaster'
        self.box_pos = torch.tensor(mjd.xpos[self.box_body], dtype=torch.float32, device=self.device)
        self.dt = 1.0 / spec['hz']
        self.deadline_steps = int(deadline_s * spec['hz'])
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.command, self.prev_action = z(n, 6), z(n, 6)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stable, self.prev_potential = z(n), z(n)
        self.held_cube, self.cube_age = z(n, 3), z(n)
        self.graph = None
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))
        self._capture()

    def _capture(self):
        wp, mjw = self.wp, self.mjw
        mjw.step(self.m, self.d)          # compile kernels outside the graph
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            for _ in range(self.substeps):
                mjw.step(self.m, self.d)
        self.graph = cap.graph
        self.reset(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def _u(self, lo, hi, k):
        return lo + (hi - lo) * torch.rand(k, generator=self.g, device=self.device)

    def reset(self, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        k = len(idx)
        if k == 0:
            return
        c = self.curriculum
        q = self.qpos[idx]
        q[:] = 0
        q[:, :6] = self.home + torch.randn(k, 6, generator=self.g, device=self.device) * 0.03 * c
        if self.cube_q == 7:
            q[:, 6] = q[:, 5]
        if self.mimic:      # ROSMASTER gripper mimic joints, as in the source URDF
            q[:, 6:11] = q[:, 5:6] * torch.tensor([1., 1., -1., -1., 1.], device=self.device)
        q[:, self.cube_q] = self._u(0.24 - 0.02 - 0.03 * c, 0.24 + 0.02 + 0.03 * c, k)
        q[:, self.cube_q + 1] = self._u(-0.06 - 0.02 - 0.03 * c, -0.06 + 0.02 + 0.03 * c, k)
        q[:, self.cube_q + 2] = 0.017
        yaw = self._u(-0.6 * c, 0.6 * c, k)
        q[:, self.cube_q + 3] = torch.cos(yaw / 2)
        q[:, self.cube_q + 6] = torch.sin(yaw / 2)
        self.qpos[idx] = q
        self.qvel[idx] = 0
        self.ctrl[idx] = q[:, :6]
        self.act[idx] = q[:, :6]
        self.command[idx] = q[:, :6]
        self.prev_action[idx] = 0
        self.steps[idx] = 0
        self.stable[idx] = 0
        self.cube_age[idx] = 0
        self.held_cube[idx] = q[:, self.cube_q:self.cube_q + 3]
        self.mjw.kinematics(self.m, self.d)
        self.prev_potential[idx] = self._potential()[idx]

    def reset_all(self, desync=False):
        self.reset(torch.ones(self.n, dtype=torch.bool, device=self.device))
        if desync:
            self.steps = torch.randint(0, self.deadline_steps // 2, (self.n,), generator=self.g, device=self.device)
        return self.observe()

    # ---------- task quantities ----------
    def cube(self):
        return self.qpos[:, self.cube_q:self.cube_q + 3]

    def gripper(self):
        return self.site_xpos[:, self.grip_site]

    def _state(self):
        cube, ee = self.cube(), self.gripper()
        box = self.box_pos
        reach = (ee - cube).norm(dim=1)
        open_grip = self.qpos[:, 5] > self.open_threshold
        lifted = (cube[:, 2] > 0.035) & (reach < 0.04)
        to_box = (cube[:, :2] - box[:2]).norm(dim=1)
        inside = ((cube[:, 0] - box[0]).abs() < 0.034) & ((cube[:, 1] - box[1]).abs() < 0.024) & (cube[:, 2] > 0.018) & (cube[:, 2] < 0.04)
        released = open_grip & (reach > 0.055)
        still = self.qvel[:, self.cube_v:self.cube_v + 3].norm(dim=1) < 0.025
        return cube, ee, reach, lifted, to_box, inside, released, still

    def _potential(self):
        cube, ee, reach, lifted, to_box, inside, released, still = self._state()
        height = (cube[:, 2] - 0.017).clamp(0, 0.06) / 0.06
        place = torch.exp(-to_box / 0.05)
        return (1 - torch.tanh(8 * reach)) + 2 * lifted.float() * height + 3 * place * (cube[:, 2] > 0.025).float() \
            + 3 * inside.float() + 2 * (inside & released).float()

    def observe(self):
        cube = self.cube()
        visible = torch.rand(self.n, generator=self.g, device=self.device) > 0.05
        noisy = cube + torch.randn(cube.shape, generator=self.g, device=self.device) * 0.005
        self.held_cube = torch.where(visible[:, None], noisy, self.held_cube)
        self.cube_age = torch.where(visible, torch.zeros_like(self.cube_age), (self.cube_age + self.dt).clamp(max=1.0))
        span = self.hi - self.lo
        norm = lambda q: 2 * (q - self.lo) / span - 1
        actor = torch.cat([norm(self.qpos[:, :6]), self.qvel[:, :6] / 3, norm(self.command), self.held_cube / 0.3,
                           visible.float()[:, None], self.cube_age[:, None], self.box_pos.expand(self.n, 3) / 0.3,
                           self.prev_action], 1)
        _, ee, reach, lifted, to_box, inside, released, still = self._state()
        critic = torch.cat([actor, cube / 0.3, ee / 0.3, (ee - cube) / 0.1, self.qvel[:, self.cube_v:self.cube_v + 6] / 3,
                            torch.stack([lifted.float(), inside.float(), released.float(), self.stable,
                                         self.steps.float() / self.deadline_steps], 1)], 1)
        return actor, critic

    def step(self, action):
        action = action.clamp(-1, 1)
        self.command = torch.minimum(torch.maximum(self.command + action * self.max_step, self.lo), self.hi)
        self.ctrl[:] = self.command
        self.wp.capture_launch(self.graph)
        self.steps += 1
        cube, ee, reach, lifted, to_box, inside, released, still = self._state()
        ok = inside & released & still
        self.stable = torch.where(ok, self.stable + self.dt, torch.zeros_like(self.stable))
        success = self.stable >= 0.5 - 1e-6
        fallen = (cube[:, 2] < -0.02) | (cube[:, :2].norm(dim=1) > 0.6)
        bad = ~torch.isfinite(self.qpos).all(1)
        pot = self._potential()
        reward = 2.0 * (0.99 * pot - self.prev_potential) + 0.05 * pot - 0.005 * (action ** 2).sum(1) \
            - 0.01 * ((action - self.prev_action) ** 2).sum(1) + 20.0 * success.float() - 2.0 * fallen.float()
        self.prev_potential = pot
        reward = torch.nan_to_num(reward)
        terminated = success | fallen | bad
        truncated = (self.steps >= self.deadline_steps) & ~terminated
        self.prev_action = action
        info = {'success': success, 'collision': fallen | bad, 'timeout': truncated, 'lifted': lifted, 'inside': inside}
        self.reset(terminated | truncated)
        obs, priv = self.observe()
        return obs, priv, reward, terminated, truncated, info

    @property
    def adim(self):
        return 6
