"""Full-MuJoCo validation environment for dock policies (M8).

Same scenario sampling, robot agent layer, drive model and sensor noise as the GPU
training environment (it subclasses it with n=1 on CPU), but:
  * motion comes from MuJoCo: the chassis surrogate of explorer3.sim.mobile (manufacturer
    geometry) tracks the drive-model velocity with finite forces, and contacts with the
    board, walls and obstacles are physical (pushing, blocking) instead of a geometric stop;
  * lidar ranges come from mujoco.mj_multiRay against the actual scene geometry.
A policy is accepted only on this environment's frozen evaluation set.
"""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from . import contract
from .dock_gpu import DockBatch, wrap

ROBOT_GROUP = 3


def scene_xml(robot, room, board_half, standoff, max_obstacles):
    from explorer3.sim.mobile import model_xml
    xml, mass, _ = model_xml(robot)
    root = ET.fromstring(xml)
    world = root.find('worldbody')
    for name in ('target',):
        node = world.find(f".//body[@name='{name}']")
        if node is not None:
            world.remove(node)
    for geom in list(world.findall('geom')):
        if geom.get('name') == 'dock_zone':
            world.remove(geom)
    base = world.find(".//body[@name='base']")
    for geom in base.iter('geom'):
        geom.set('group', str(ROBOT_GROUP))
    ET.SubElement(world, 'geom', name='board', type='box', pos=f'-0.015 0 0.2', size=f'0.015 {board_half} 0.2',
                  rgba='.85 .3 .12 1', contype='2', conaffinity='2')
    ET.SubElement(world, 'geom', name='zone', type='cylinder', pos=f'{standoff} 0 .002', size='.08 .002',
                  contype='0', conaffinity='0', rgba='.15 .65 .35 1')
    x0, y0, x1, y1 = room
    for i, (cx, cy, hx, hy) in enumerate([(x0 - .05, 0, .05, (y1 - y0) / 2 + .1), (x1 + .05, 0, .05, (y1 - y0) / 2 + .1),
                                           ((x0 + x1) / 2, y0 - .05, (x1 - x0) / 2, .05), ((x0 + x1) / 2, y1 + .05, (x1 - x0) / 2, .05)]):
        ET.SubElement(world, 'geom', name=f'wall{i}', type='box', pos=f'{cx} {cy} .2', size=f'{hx} {hy} .2',
                      rgba='.7 .7 .72 1', contype='2', conaffinity='2')
    for j in range(max_obstacles):
        body = ET.SubElement(world, 'body', name=f'obstacle{j}', mocap='true', pos=f'0 0 -5')
        ET.SubElement(body, 'geom', name=f'obstacle{j}', type='box', size='.05 .05 .15', rgba='.35 .45 .6 1',
                      contype='2', conaffinity='2')
    for motor in root.find('actuator'):
        motor.set('ctrlrange', '-60 60' if motor.get('joint') in ('x', 'y') else '-12 12')
    return ET.tostring(root, encoding='unicode'), mass


class DockMujoco(DockBatch):
    physics_substeps_check = True

    def __init__(self, robot='tank', seed=0, curriculum=1.0, scenario='dock_v1'):
        self._ready = False
        super().__init__(robot, n=1, device='cpu', seed=seed, scenario=scenario, curriculum=curriculum)
        xml, self.mass = scene_xml(robot, self.room, self.board_half, self.standoff, self.max_obstacles)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body('base').id
        self.obstacle_geoms = [self.model.geom(f'obstacle{j}').id for j in range(self.max_obstacles)]
        self.obstacle_bodies = [self.model.body(f'obstacle{j}').mocapid[0] for j in range(self.max_obstacles)]
        self.robot_geoms = {i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] != 0 and
                            self.model.geom_group[i] == ROBOT_GROUP}
        self.scene_geoms = {self.model.geom(n).id for n in ['board'] + [f'wall{i}' for i in range(4)]} | set(self.obstacle_geoms)
        self.substeps = round(self.dt / self.model.opt.timestep)
        self._ready = True
        self.reset_all()

    def reset(self, mask):
        super().reset(mask)
        if not getattr(self, '_ready', False) or not bool(mask.any()):
            return
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.pose[0].numpy()
        for j, (gid, mid) in enumerate(zip(self.obstacle_geoms, self.obstacle_bodies)):
            box = self.boxes[0, 4 + j].numpy()
            if bool(self.box_active[0, 4 + j]):
                self.data.mocap_pos[mid] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, .15]
                self.model.geom_size[gid] = [(box[2] - box[0]) / 2, (box[3] - box[1]) / 2, .15]
            else:
                self.data.mocap_pos[mid] = [0, 0, -5]
        mujoco.mj_forward(self.model, self.data)

    def lidar_bins(self, noisy=True):
        if not getattr(self, '_ready', False):
            return super().lidar_bins(noisy)
        yaw = float(self.pose[0, 2])
        origin = np.array([self.pose[0, 0] + self.lidar_forward * math.cos(yaw),
                           self.pose[0, 1] + self.lidar_forward * math.sin(yaw), 0.18])
        angles = yaw + self.ray_angles.numpy()
        vec = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], 1).ravel()
        n = len(angles)
        geomid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        group = np.array([1, 1, 1, 0, 1, 1], np.uint8)   # exclude the robot's own geoms (group 3)
        mujoco.mj_multiRay(self.model, self.data, origin, vec, group, 1, -1, geomid, dist, None, n, 8.0)
        ranges = torch.tensor(np.where(dist < 0, 8.0, dist), dtype=torch.float32)[None]
        if noisy:
            ranges = ranges + torch.randn(ranges.shape, generator=self.g) * (self.params['lidar_noise'][:, None] * (1 + ranges))
        bins = ranges.view(1, contract.LIDAR_BINS, self.rays_per_bin).min(-1).values
        if noisy:
            drop = torch.rand(bins.shape, generator=self.g) < self.params['dropout'][:, None]
            bins = torch.where(drop, torch.zeros_like(bins), bins)
        return torch.where(bins > contract.LIDAR_MAX_M, torch.zeros_like(bins), bins)

    def integrate(self):
        """MuJoCo dynamics: finite forces track the drive-model velocity; contacts are physical."""
        target_body = self.vel[0].numpy().copy()
        collided = False
        for _ in range(self.substeps):
            yaw = self.data.qpos[2]
            c, s = math.cos(yaw), math.sin(yaw)
            desired = np.array([c * target_body[0] - s * target_body[1], s * target_body[0] + c * target_body[1], target_body[2]])
            gain = np.array([self.mass / 0.02, self.mass / 0.02, 0.3])
            self.data.ctrl[:3] = np.clip(gain * (desired - self.data.qvel[:3]), self.model.actuator_ctrlrange[:3, 0],
                                         self.model.actuator_ctrlrange[:3, 1])
            mujoco.mj_step(self.model, self.data)
            for k in range(self.data.ncon):
                g1, g2 = self.data.contact[k].geom1, self.data.contact[k].geom2
                if (g1 in self.robot_geoms and g2 in self.scene_geoms) or (g2 in self.robot_geoms and g1 in self.scene_geoms):
                    collided = True
        q, v = self.data.qpos[:3], self.data.qvel[:3]
        yaw = q[2]
        c, s = math.cos(yaw), math.sin(yaw)
        self.pose = torch.tensor([[q[0], q[1], math.atan2(math.sin(yaw), math.cos(yaw))]], dtype=torch.float32)
        self.vel = torch.tensor([[c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]], dtype=torch.float32)
        return torch.tensor([collided])

    def collided(self):
        return torch.zeros(1, dtype=torch.bool)


def evaluate(policy, robot, seeds, curriculum=1.0, max_steps=None):
    """Run a numpy Policy on frozen seeds in full MuJoCo. Returns per-episode records."""
    env = DockMujoco(robot, seed=0, curriculum=curriculum)
    out = []
    for seed in seeds:
        env.g.manual_seed(int(seed))
        env.reset(torch.ones(1, dtype=torch.bool))
        obs, _ = env.observe()
        policy.reset()
        record = {'seed': int(seed), 'outcome': 'timeout', 'steps': 0, 'safety_stops': 0}
        for t in range(max_steps or env.deadline_steps):
            a = torch.tensor(policy.act(obs[0].numpy()))[None]
            action = env.command(a, env.lidar_bins(noisy=True))
            collided = env.integrate()
            _, term, trunc, info = env.finish(action, collided, auto_reset=False)
            record['safety_stops'] += int(info['safety_stop'][0])
            record['steps'] = t + 1
            if bool(info['success'][0]):
                record['outcome'] = 'success'
                break
            if bool(info['collision'][0]):
                record['outcome'] = 'collision'
                break
            if bool(trunc[0]):
                break
            obs, _ = env.observe()
        record['final_distance'] = float(info['distance'][0])
        record['final_yaw_error'] = float(info['yaw_error'][0])
        out.append(record)
    return out
