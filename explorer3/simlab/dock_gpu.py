"""Batched GPU training environment for the dock task (Tank, ROSMASTER base).

Planar rigid-body model with the same equations as the MuJoCo chassis surrogate
(explorer3.sim.mobile), extended with what the real robots do and what the MuJoCo
surrogate lacked:
  * drive: command delay (steps), dead zone, first-order response, slip (per world);
  * the robot agent's own layer: hard clamp, acceleration limit, lidar stop
    (forward < 0.15 m -> <= 0.1 m/s, < 0.05 m -> 0), identical to the real agents;
  * sensors in the contract format (explorer3.simlab.contract): 36-sector lidar by ray
    casting against the scene boxes, IMU (gyro z, planar specific force), ArUco-like
    target detection with field of view, occlusion, noise, dropouts and bursts.
Policies are validated afterwards in full MuJoCo (explorer3.simlab.dock_mujoco).

Frame: target frame. Target board occupies x in [-0.03, 0], marker normal +x, the stop
zone centre is (standoff, 0); the docked heading is pi (facing the board).
"""
import math

import torch

from . import contract
from .scenario import load_measurements, load_scenario

TWO_PI = 2 * math.pi


def wrap(a):
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


class DockBatch:
    def __init__(self, robot='tank', n=4096, device='cuda', seed=0, scenario='dock_v1', curriculum=0.0,
                 rays_per_bin=4, max_obstacles=3):
        if robot not in contract.ROBOTS:
            raise ValueError('Unknown robot')
        self.robot, self.n, self.device = robot, n, torch.device(device)
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        self.sc = load_scenario(scenario)
        self.meas = load_measurements()
        spec = contract.ROBOTS[robot]
        self.adim = len(spec['action'])
        self.dt = 1.0 / spec['control_hz']
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=self.device)
        self.policy_limits = f(spec['policy_limits'])
        self.hard_limits = f(spec['hard_limits'])
        self.accel_limits = f(spec['accel_limits'])
        m = self.meas[robot]
        l, w = m['footprint_lw']
        self.radius = 0.5 * math.hypot(l, w)
        self.cam_forward, self.hfov = m['camera']['forward'], m['camera']['hfov_rad']
        self.lidar_forward = m['lidar']['forward']
        target = self.meas['dock_target']
        self.standoff = target['standoff']
        self.board_half = target['board_size_wh'][0] / 2
        room = self.meas['room']['free_area_xy']
        self.room = (-0.1, -room[1] / 2, room[0] - 0.1, room[1] / 2)
        self.curriculum = curriculum
        self.max_obstacles = max_obstacles
        self.deadline_steps = int(self.sc['deadline_s'] * spec['control_hz'])
        rays = contract.LIDAR_BINS * rays_per_bin
        self.rays_per_bin = rays_per_bin
        self.ray_angles = (torch.arange(rays, device=self.device) + 0.5) * TWO_PI / rays - math.pi / contract.LIDAR_BINS
        self.max_delay = 4
        z = lambda *s: torch.zeros(*s, device=self.device)
        N = n
        self.pose, self.vel = z(N, 3), z(N, 3)            # world (target frame) x, y, yaw; body vx, vy, w
        self.cmd_queue = z(N, self.max_delay, 3)
        self.agent_out = z(N, 3)                          # after clamp/accel/lidar stop
        self.prev_action = z(N, self.adim)
        self.steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self.stable = z(N)
        self.boxes = z(N, 4 + max_obstacles + 1, 4)       # walls(4), obstacles, board
        self.box_active = torch.zeros(N, 4 + max_obstacles + 1, dtype=torch.bool, device=self.device)
        self.held = z(N, 4)                               # distance, bearing, yaw, age
        self.burst = z(N)
        self.prev_goal_dist = z(N)
        self.prev_body_vel = z(N, 3)
        self.params = {k: z(N) for k in ('delay', 'tau', 'slip_v', 'slip_w', 'dz_v', 'dz_w', 'gyro_bias',
                                         'imu_noise', 'lidar_noise', 'dropout')}
        self.safety_events = z(N)
        self.reset_all()

    # ---------- sampling ----------
    def _u(self, lo, hi, shape=None):
        shape = shape or (self.n,)
        return lo + (hi - lo) * torch.rand(shape, generator=self.g, device=self.device)

    def _range(self, key):
        full, easy = self.sc['starts'][key], self.sc['curriculum'][key]
        c = self.curriculum
        return easy[0] + c * (full[0] - easy[0]), easy[1] + c * (full[1] - easy[1])

    def reset(self, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        k = len(idx)
        if k == 0:
            return
        u = lambda lo, hi: self._u(lo, hi, (k,))
        d = u(*self._range('distance_from_zone'))
        b = u(*self._range('bearing_from_normal'))
        he = u(*self._range('heading_error'))
        x = self.standoff + d * torch.cos(b)
        y = d * torch.sin(b)
        x = x.clamp(self.room[0] + self.radius + 0.35, self.room[2] - self.radius - 0.05)
        y = y.clamp(self.room[1] + self.radius + 0.05, self.room[3] - self.radius - 0.05)
        toward = torch.atan2(-y, self.standoff - x)
        self.pose[idx] = torch.stack([x, y, wrap(toward + he)], 1)
        self.vel[idx] = 0
        self.prev_body_vel[idx] = 0
        self.cmd_queue[idx] = 0
        self.agent_out[idx] = 0
        self.prev_action[idx] = 0
        self.steps[idx] = 0
        self.stable[idx] = 0
        self.burst[idx] = 0
        self.held[idx] = torch.tensor([0., 0., 0., contract.AGE_CLIP_S], device=self.device)
        p = self.params
        p['delay'][idx] = torch.randint(1, self.max_delay, (k,), generator=self.g, device=self.device).float()
        p['tau'][idx] = u(0.05, 0.18)
        p['slip_v'][idx] = u(0.80, 1.0)
        p['slip_w'][idx] = u(0.65, 1.0) if self.robot == 'tank' else u(0.85, 1.0)
        p['dz_v'][idx] = u(0.0, 0.03)
        p['dz_w'][idx] = u(0.0, 0.12)
        p['gyro_bias'][idx] = u(-0.02, 0.02)
        p['imu_noise'][idx] = u(0.05, 0.4)
        p['lidar_noise'][idx] = u(0.005, 0.02)
        p['dropout'][idx] = u(0.0, 2 * self.sc['disturbances']['lidar_bin_dropout_p'])
        # scene boxes: walls, board, obstacles (axis-aligned in the target frame)
        r = self.room
        walls = torch.tensor([[r[0] - 0.1, r[1] - 0.1, r[0], r[3] + 0.1], [r[2], r[1] - 0.1, r[2] + 0.1, r[3] + 0.1],
                              [r[0], r[1] - 0.1, r[2], r[1]], [r[0], r[3], r[2], r[3] + 0.1]], device=self.device)
        self.boxes[idx, :4] = walls
        self.box_active[idx, :4] = True
        self.boxes[idx, -1] = torch.tensor([-0.03, -self.board_half, 0.0, self.board_half], device=self.device)
        self.box_active[idx, -1] = True
        lo_n, hi_n = self.sc['obstacles']['count']
        top = lo_n + (hi_n - lo_n) * self.curriculum
        count = torch.floor(self._u(0.0, top + 1.0, (k,))).clamp(max=min(top, self.max_obstacles))
        (sx0, sy0), (sx1, sy1) = self.sc['obstacles']['size_xy']
        clear = self.sc['obstacles']['keep_clear_of_zone']
        for j in range(self.max_obstacles):
            cx, cy = u(0.3, r[2] - 0.3), u(r[1] + 0.3, r[3] - 0.3)
            hx, hy = u(sx0, sx1) / 2, u(sy0, sy1) / 2
            box = torch.stack([cx - hx, cy - hy, cx + hx, cy + hy], 1)
            near_zone = self._box_dist(box, torch.full((k,), self.standoff, device=self.device), torch.zeros(k, device=self.device)) < clear + self.radius
            near_start = self._box_dist(box, x, y) < self.radius + 0.15
            self.boxes[idx, 4 + j] = box
            self.box_active[idx, 4 + j] = (j < count) & ~near_zone & ~near_start
        self.prev_goal_dist[idx] = self._goal_error()[0][idx]

    def reset_all(self):
        self.reset(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self.observe()

    @staticmethod
    def _box_dist(box, x, y):
        dx = torch.maximum(torch.maximum(box[..., 0] - x, x - box[..., 2]), torch.zeros_like(x))
        dy = torch.maximum(torch.maximum(box[..., 1] - y, y - box[..., 3]), torch.zeros_like(y))
        return torch.hypot(dx, dy)

    # ---------- geometry ----------
    def _cast(self, origin, angles, max_range=8.0):
        """origin (N,2), angles (N,R) world -> distance to the nearest active box (N,R)."""
        dx, dy = torch.cos(angles), torch.sin(angles)
        ox, oy = origin[:, 0:1, None], origin[:, 1:2, None]
        b = self.boxes[:, None]                                       # N,1,K,4
        inv_x = 1.0 / torch.where(dx.abs() < 1e-9, torch.full_like(dx, 1e-9), dx)[..., None]
        inv_y = 1.0 / torch.where(dy.abs() < 1e-9, torch.full_like(dy, 1e-9), dy)[..., None]
        t1, t2 = (b[..., 0] - ox) * inv_x, (b[..., 2] - ox) * inv_x
        t3, t4 = (b[..., 1] - oy) * inv_y, (b[..., 3] - oy) * inv_y
        tmin = torch.maximum(torch.minimum(t1, t2), torch.minimum(t3, t4))
        tmax = torch.minimum(torch.maximum(t1, t2), torch.maximum(t3, t4))
        hit = (tmax >= torch.clamp(tmin, min=0)) & self.box_active[:, None]
        t = torch.where(hit, torch.clamp(tmin, min=0), torch.full_like(tmin, max_range))
        return t.min(-1).values.clamp(max=max_range)

    def _goal_error(self):
        dx = self.pose[:, 0] - self.standoff
        dy = self.pose[:, 1]
        return torch.hypot(dx, dy), wrap(self.pose[:, 2] - math.pi)

    def collided(self):
        x, y = self.pose[:, 0:1], self.pose[:, 1:2]
        d = self._box_dist(self.boxes, x, y)
        return ((d < self.radius) & self.box_active).any(1)

    # ---------- sensors ----------
    def lidar_bins(self, noisy=True):
        yaw = self.pose[:, 2:3]
        origin = self.pose[:, :2] + self.lidar_forward * torch.cat([torch.cos(yaw), torch.sin(yaw)], 1)
        ranges = self._cast(origin, yaw + self.ray_angles[None])
        if noisy:
            ranges = ranges + torch.randn(ranges.shape, generator=self.g, device=self.device) * (
                self.params['lidar_noise'][:, None] * (1 + ranges))
        bins = ranges.view(self.n, contract.LIDAR_BINS, self.rays_per_bin).min(-1).values
        if noisy:
            drop = torch.rand(bins.shape, generator=self.g, device=self.device) < self.params['dropout'][:, None]
            bins = torch.where(drop, torch.zeros_like(bins), bins)
        bins = torch.where(bins > contract.LIDAR_MAX_M, torch.zeros_like(bins), bins)   # beyond range -> no return
        return bins

    def detect_target(self):
        yaw = self.pose[:, 2]
        cam = self.pose[:, :2] + self.cam_forward * torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
        rel = -cam                                   # marker centre at the origin
        dist = rel.norm(dim=1)
        bearing = wrap(torch.atan2(rel[:, 1], rel[:, 0]) - yaw)
        facing = cam[:, 0] > 0.05                    # in front of the board
        view_angle = torch.atan2(cam[:, 1].abs(), cam[:, 0])
        los = self._cast(cam, torch.atan2(rel[:, 1], rel[:, 0])[:, None])[:, 0] >= dist - 0.04
        dist_ok = (dist > 0.12) & (dist < 3.0)
        dis = self.sc['disturbances']
        start_burst = torch.rand(self.n, generator=self.g, device=self.device) < dis['marker_burst_loss_p']
        self.burst = torch.where(start_burst & (self.burst <= 0), self._u(0.5, 1.5), torch.clamp(self.burst - self.dt, min=0))
        drop = torch.rand(self.n, generator=self.g, device=self.device) < dis['marker_dropout_p']
        visible = (bearing.abs() < 0.45 * self.hfov) & facing & (view_angle < 1.2) & los & dist_ok & ~drop & (self.burst <= 0)
        noise = lambda s: torch.randn(self.n, generator=self.g, device=self.device) * s
        meas = torch.stack([dist + noise(0.01 + 0.02 * dist), bearing + noise(0.01), wrap(yaw - math.pi + noise(0.05))], 1)
        self.held[:, :3] = torch.where(visible[:, None], meas, self.held[:, :3])
        self.held[:, 3] = torch.where(visible, torch.zeros_like(dist), (self.held[:, 3] + self.dt).clamp(max=contract.AGE_CLIP_S))
        return visible

    def imu(self):
        accel = (self.vel - self.prev_body_vel) / self.dt
        ax = accel[:, 0] - self.vel[:, 2] * self.vel[:, 1]
        ay = accel[:, 1] + self.vel[:, 2] * self.vel[:, 0]
        noise = lambda s: torch.randn(self.n, generator=self.g, device=self.device) * s
        s = self.params['imu_noise']
        return torch.stack([self.vel[:, 2] + self.params['gyro_bias'] + noise(0.01), ax + noise(s), ay + noise(s)], 1)

    def observe(self):
        visible = self.detect_target()
        lidar = self.lidar_bins()
        h = self.held
        obs = torch.cat([
            torch.stack([h[:, 0].clamp(max=2 * contract.TARGET_SCALE_M) / contract.TARGET_SCALE_M,
                         torch.sin(h[:, 1]), torch.cos(h[:, 1]), h[:, 2] / math.pi,
                         visible.float(), h[:, 3] / contract.AGE_CLIP_S], 1),
            torch.where(lidar <= 0, torch.ones_like(lidar), lidar.clamp(max=contract.LIDAR_MAX_M) / contract.LIDAR_MAX_M),
            self.imu() / 2.0, self.prev_action], 1)
        return obs, self.privileged()

    def privileged(self):
        dist, yaw_err = self._goal_error()
        yaw = self.pose[:, 2]
        dx, dy = self.standoff - self.pose[:, 0], -self.pose[:, 1]
        c, s = torch.cos(yaw), torch.sin(yaw)
        p = self.params
        true_lidar = self.lidar_bins(noisy=False)
        true_lidar = torch.where(true_lidar <= 0, torch.ones_like(true_lidar), true_lidar / contract.LIDAR_MAX_M)
        return torch.cat([torch.stack([c * dx + s * dy, -s * dx + c * dy, torch.sin(yaw_err), torch.cos(yaw_err),
                                       self.stable, self.steps.float() / self.deadline_steps,
                                       p['delay'] / self.max_delay, p['tau'] * 5, p['slip_v'], p['slip_w'],
                                       p['dz_v'] * 10, p['dz_w'] * 5], 1),
                          self.vel, true_lidar, self.prev_action], 1)

    # ---------- dynamics ----------
    def _agent_layer(self, cmd, lidar):
        """Real agent behaviour: clamp, acceleration limit, lidar stop. cmd (N,3) physical units."""
        cmd = torch.maximum(torch.minimum(cmd, self._lim(self.hard_limits)), -self._lim(self.hard_limits))
        step = self._lim(self.accel_limits) * self.dt
        cmd = self.agent_out + torch.maximum(torch.minimum(cmd - self.agent_out, step), -step)
        nb = contract.LIDAR_BINS
        front_idx = torch.tensor([i % nb for i in range(-3, 4)], device=self.device)
        rear_idx = torch.tensor([(nb // 2 + i) % nb for i in range(-3, 4)], device=self.device)
        lid = torch.where(lidar <= 0, torch.full_like(lidar, 99.), lidar)
        front, rear = lid[:, front_idx].min(1).values, lid[:, rear_idx].min(1).values
        v = cmd[:, 0]
        cap_f = torch.where(front < 0.05, torch.zeros_like(v), torch.where(front < 0.15, torch.full_like(v, 0.1), torch.full_like(v, 99.)))
        cap_r = torch.where(rear < 0.05, torch.zeros_like(v), torch.where(rear < 0.15, torch.full_like(v, 0.1), torch.full_like(v, 99.)))
        limited = torch.where(v > 0, torch.minimum(v, cap_f), torch.maximum(v, -cap_r))
        self.safety_events = (limited != v).float()
        cmd = torch.cat([limited[:, None], cmd[:, 1:]], 1)
        self.agent_out = cmd
        return cmd

    def _lim(self, limits):
        if self.robot == 'tank':
            return torch.stack([limits[0], torch.zeros((), device=self.device), limits[1]])
        return limits

    def command(self, action, lidar):
        """Policy action -> robot agent layer -> link delay -> drive response. Updates self.vel (body)."""
        action = action.clamp(-1, 1)
        phys = action * self.policy_limits
        cmd = phys if self.robot != 'tank' else torch.stack([phys[:, 0], torch.zeros_like(phys[:, 0]), phys[:, 1]], 1)
        cmd = self._agent_layer(cmd, lidar)
        self.cmd_queue = torch.roll(self.cmd_queue, 1, dims=1)
        self.cmd_queue[:, 0] = cmd
        delayed = self.cmd_queue[torch.arange(self.n, device=self.device), self.params['delay'].long()]
        p = self.params
        dz = torch.stack([p['dz_v'], p['dz_v'], p['dz_w']], 1)
        delayed = torch.where(delayed.abs() < dz, torch.zeros_like(delayed), delayed)
        slip = torch.stack([p['slip_v'], p['slip_v'], p['slip_w']], 1)
        alpha = (1 - torch.exp(-self.dt / p['tau']))[:, None]
        self.prev_body_vel = self.vel.clone()
        self.vel = self.vel + alpha * (slip * delayed - self.vel)
        if self.robot == 'tank':
            self.vel[:, 1] = 0.02 * self.vel[:, 2] * self.vel[:, 0]   # small skid during turning
        return action

    def integrate(self):
        collided = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        for _ in range(5):
            yaw = self.pose[:, 2]
            c, s = torch.cos(yaw), torch.sin(yaw)
            vx, vy, w = self.vel.unbind(1)
            old = self.pose.clone()
            self.pose = self.pose + torch.stack([c * vx - s * vy, s * vx + c * vy, w], 1) * (self.dt / 5)
            self.pose[:, 2] = wrap(self.pose[:, 2])
            hit = self.collided()
            self.pose = torch.where(hit[:, None], old, self.pose)
            collided |= hit
        self.vel = torch.where(collided[:, None], torch.zeros_like(self.vel), self.vel)
        return collided

    def finish(self, action, collided, auto_reset=True):
        self.steps += 1
        dist, yaw_err = self._goal_error()
        su = self.sc['success']
        speed = torch.hypot(self.vel[:, 0], self.vel[:, 1])
        in_zone = (dist < su['position_tolerance']) & (yaw_err.abs() < su['yaw_tolerance']) & (speed < su['stationary_speed']) & (self.vel[:, 2].abs() < 0.05)
        self.stable = torch.where(in_zone, self.stable + self.dt, torch.zeros_like(self.stable))
        success = self.stable >= su['hold_s'] - 1e-6
        progress = self.prev_goal_dist - dist
        self.prev_goal_dist = dist
        near = (dist < 0.3).float()
        reward = (2.0 * progress - 0.05 * near * yaw_err.abs() - 0.01 * ((action - self.prev_action) ** 2).sum(1)
                  - 0.005 - 0.05 * self.safety_events - 5.0 * collided.float() + 10.0 * success.float())
        terminated = success | collided
        truncated = (self.steps >= self.deadline_steps) & ~terminated
        self.prev_action = action
        info = {'success': success, 'collision': collided, 'timeout': truncated, 'distance': dist,
                'yaw_error': yaw_err, 'safety_stop': self.safety_events.bool()}
        if auto_reset:
            self.reset(terminated | truncated)
        return reward, terminated, truncated, info

    def step(self, action):
        action = self.command(action, self.lidar_bins(noisy=True))
        collided = self.integrate()
        reward, terminated, truncated, info = self.finish(action, collided)
        obs, priv = self.observe()
        return obs, priv, reward, terminated, truncated, info
