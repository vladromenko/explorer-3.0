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


def render_option():
    """Robot geoms live in group 3 (excluded from lidar rays) but must be drawn in videos."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    return opt


def marker_texture(marker_id=0, pixels=600):
    """Printable ArUco target (white margin) used both on the simulated board and for printing."""
    import cv2
    from explorer3.paths import DATA
    path = DATA / 'simlab' / 'assets' / f'aruco_4x4_{marker_id}.png'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        inner = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, pixels)
        img = cv2.copyMakeBorder(inner, pixels // 8, pixels // 8, pixels // 8, pixels // 8, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def scene_xml(robot, room, board_half, standoff, max_obstacles):
    """Living-room style scene: laminate floor, low light walls, white target board with a real
    ArUco marker, green stop zone, cardboard obstacles, warm key light with soft shadows."""
    from explorer3.sim.mobile import model_xml
    from .scenario import load_measurements
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
        if geom.get('material') is None:
            geom.set('rgba', '.24 .26 .29 1' if robot == 'tank' else '.20 .22 .25 1')
    # ---- look
    asset = root.find('asset')
    for tex in list(asset.findall('texture')):
        if tex.get('name') in ('explorer_sky', 'explorer_grid'):
            asset.remove(tex)
    for mat in list(asset.findall('material')):
        if mat.get('name') == 'explorer_floor':
            asset.remove(mat)
    ET.SubElement(asset, 'texture', name='explorer_sky', type='skybox', builtin='gradient', rgb1='.32 .34 .38', rgb2='.10 .11 .13', width='256', height='1536')
    ET.SubElement(asset, 'texture', name='laminate', type='2d', builtin='checker', rgb1='.56 .45 .35', rgb2='.52 .41 .32', width='512', height='512')
    ET.SubElement(asset, 'material', name='explorer_floor', texture='laminate', texrepeat='.8 5', texuniform='true', reflectance='.04', specular='.2', shininess='.3')
    ET.SubElement(asset, 'material', name='outside', rgba='.16 .17 .19 1', specular='0')
    ET.SubElement(asset, 'material', name='wall', rgba='.88 .87 .84 1', specular='.05')
    ET.SubElement(asset, 'material', name='board', rgba='.97 .97 .96 1', specular='.1')
    ET.SubElement(asset, 'material', name='cardboard', rgba='.70 .55 .38 1', specular='.05')
    ET.SubElement(asset, 'texture', name='aruco', type='2d', file=str(marker_texture(load_measurements()['dock_target']['marker']['id'])))
    ET.SubElement(asset, 'material', name='aruco', texture='aruco', specular='.05')
    visual = root.find('visual')
    for child in list(visual):
        if child.tag in ('headlight', 'rgba', 'map', 'quality'):
            visual.remove(child)
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.25 .25 .25', specular='.02 .02 .02')
    ET.SubElement(visual, 'quality', shadowsize='4096', offsamples='8')
    ET.SubElement(visual, 'map', znear='.01', zfar='30', shadowclip='3', shadowscale='.5')
    for light in world.findall('light'):
        world.remove(light)
    ET.SubElement(world, 'light', name='key', pos='1.5 -1.2 3', dir='-.3 .3 -1', directional='true', diffuse='.75 .70 .62', specular='.2 .2 .2', castshadow='true')
    ET.SubElement(world, 'light', name='fill', pos='-1 1 2.5', dir='.3 -.2 -1', directional='true', diffuse='.25 .28 .32', castshadow='false')
    x0, y0, x1, y1 = room
    floor = world.find("geom[@name='floor']")
    floor.attrib.pop('material', None); floor.set('rgba', '.13 .14 .16 1'); floor.set('size', '8 8 .05'); floor.set('pos', '0 0 -.02')
    ET.SubElement(world, 'geom', name='room_floor', type='box', pos=f'{(x0 + x1) / 2} {(y0 + y1) / 2} -.0005',
                  size=f'{(x1 - x0) / 2} {(y1 - y0) / 2} .0005', material='explorer_floor', contype='0', conaffinity='0')
    stat = root.find('statistic')
    if stat is not None:
        stat.set('center', f'{(x0 + x1) / 2} 0 .2'); stat.set('extent', f'{max(x1 - x0, y1 - y0) * .75}')
    # ---- target board with the marker, zone ring
    target = load_measurements()['dock_target']
    height = target['board_size_wh'][1]
    side = target['marker']['side'] * 1.25          # printed sheet incl. white margin
    ET.SubElement(world, 'geom', name='board', type='box', pos=f'-0.015 0 {height / 2}', size=f'0.015 {board_half} {height / 2}',
                  material='board', contype='2', conaffinity='2')
    ET.SubElement(world, 'geom', name='marker', type='box', pos=f'0.001 0 {target["marker"]["center_height"]}',
                  size=f'0.001 {side / 2} {side / 2}', euler='0 0 0', material='aruco', contype='0', conaffinity='0')
    ET.SubElement(world, 'geom', name='zone', type='cylinder', pos=f'{standoff} 0 .001', size=f'{target["zone_radius"]} .001',
                  contype='0', conaffinity='0', rgba='.18 .72 .45 .55')
    for i, (cx, cy, hx, hy) in enumerate([(x0 - .05, 0, .05, (y1 - y0) / 2 + .1), (x1 + .05, 0, .05, (y1 - y0) / 2 + .1),
                                           ((x0 + x1) / 2, y0 - .05, (x1 - x0) / 2, .05), ((x0 + x1) / 2, y1 + .05, (x1 - x0) / 2, .05)]):
        ET.SubElement(world, 'geom', name=f'wall{i}', type='box', pos=f'{cx} {cy} .2', size=f'{hx} {hy} .2',
                      material='wall', contype='2', conaffinity='2')
    for j in range(max_obstacles):
        body = ET.SubElement(world, 'body', name=f'obstacle{j}', mocap='true', pos='0 0 -5')
        ET.SubElement(body, 'geom', name=f'obstacle{j}', type='box', size='.05 .05 .15', material='cardboard',
                      contype='2', conaffinity='2')
    # ---- cameras for videos: chase (follows the robot) and overview
    # high corner camera that keeps the robot in frame wherever it drives
    ET.SubElement(world, 'camera', name='chase', mode='targetbody', target='base', pos=f'{x1 - .1} {y0 + .2} 1.25', fovy='50')
    ET.SubElement(world, 'camera', name='overview', pos=f'{(x0 + x1) / 2} {y0 - .9} 2.4', xyaxes='1 0 0 0 .62 .78', fovy='60')
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
