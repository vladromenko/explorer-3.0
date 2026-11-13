"""Seeded episode randomization using native MuJoCo model parameters.

Protocol adaptation of Tobin 2017 and Peng 2018, not their reproduction.
Ranges are explicit software stress assumptions pending real calibration.
"""
import copy
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

RECIPE={'version':1,'status':'assumed_not_measured','modes':['nominal','varied'],
        'floor':['checker_10cm','plain_dark','plain_light'],
        'light_multiplier':[.85,1.15],'camera_translation_m':.002,'camera_rotation_degrees':.8,
        'camera_fov_delta_degrees':1.,'cube_mass_multiplier':[.85,1.15],
        'cube_sliding_friction_multiplier':[.85,1.15],
        'constraints':'Do not randomize solver tolerances, collision geometry, joint signs or calibrated units',
        'references':['https://arxiv.org/abs/1703.06907','https://arxiv.org/abs/1710.06537']}


class EpisodeVariation:
    fields=('geom_rgba','geom_matid','geom_friction','mat_rgba','light_diffuse','light_ambient',
            'cam_pos','cam_quat','cam_fovy','cam_intrinsic','body_mass','body_inertia')

    def __init__(self, model, camera):
        self.model=model;self.camera=model.camera(camera).id
        self.base={key:getattr(model,key).copy() for key in self.fields}
        self.realization={'mode':'nominal','recipe':RECIPE}

    def apply(self,seed,mode):
        if mode not in RECIPE['modes']:raise ValueError('Unknown scene variation')
        for name,value in self.base.items():getattr(self.model,name)[:]=value
        rng=np.random.default_rng(np.random.SeedSequence([seed,22026]))
        self.realization={'mode':mode,'seed':seed,'recipe':copy.deepcopy(RECIPE)}
        m=self.model
        if mode=='varied':
            floor=int(rng.integers(0,3));gid=m.geom('floor').id
            m.geom_matid[gid]=m.mat('explorer_floor' if floor==0 else 'explorer_plain').id
            if floor:
                tint=rng.uniform(.12,.23) if floor==1 else rng.uniform(.35,.5)
                m.mat_rgba[m.mat('explorer_plain').id,:3]=[tint,tint*1.04,tint*1.1]
            light=float(rng.uniform(*RECIPE['light_multiplier']));m.light_diffuse[:]*=light
            translation=rng.uniform(-.002,.002,3);angles=rng.uniform(-.8,.8,3)
            m.cam_pos[self.camera]+=translation
            rotation=Rotation.from_quat(m.cam_quat[self.camera][[1,2,3,0]])*Rotation.from_euler('xyz',angles,degrees=True)
            m.cam_quat[self.camera]=rotation.as_quat()[[3,0,1,2]]
            fov=float(rng.uniform(-1,1));old_fovy=m.cam_fovy[self.camera];m.cam_fovy[self.camera]+=fov
            if m.cam_sensorsize[self.camera,1]>0:
                # Intrinsic cameras ignore fovy during rendering; vary their
                # focal lengths too, keeping the measured principal point.
                scale=np.tan(np.deg2rad(old_fovy)/2)/np.tan(np.deg2rad(old_fovy+fov)/2)
                m.cam_intrinsic[self.camera,:2]*=scale
            self.realization.update(floor=RECIPE['floor'][floor],light_multiplier=light,
                camera_translation_m=translation.tolist(),camera_rotation_degrees=angles.tolist(),camera_fov_delta_degrees=fov)
            cube=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'task_cube')
            if cube>=0:
                mass=float(rng.uniform(*RECIPE['cube_mass_multiplier']))
                friction=float(rng.uniform(*RECIPE['cube_sliding_friction_multiplier']))
                m.body_mass[cube]*=mass;m.body_inertia[cube]*=mass
                m.geom_friction[m.geom('cube').id,0]*=friction
                self.realization.update(cube_mass_multiplier=mass,cube_sliding_friction_multiplier=friction)
        mujoco.mj_setConst(m,mujoco.MjData(m))
        return copy.deepcopy(self.realization)

    def snapshot(self):
        return {'model_parameters':{key:getattr(self.model,key).copy() for key in self.fields},
                'realization':copy.deepcopy(self.realization)}

    def restore(self,state):
        for name,value in state['model_parameters'].items():getattr(self.model,name)[:]=value
        self.realization=copy.deepcopy(state['realization'])
        mujoco.mj_setConst(self.model,mujoco.MjData(self.model))
