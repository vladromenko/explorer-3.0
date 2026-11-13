"""Native MuJoCo/Gymnasium SO-101 task. State teacher and RGB student stay separate."""
import copy
import random
import numpy as np
import gymnasium as gym
import mujoco
from .models import so101_model


class SO101PickPlace(gym.Env):
    metadata = {'render_modes':['rgb_array'], 'render_fps':30}
    fps=30
    profile='so101_pick_place_sim'
    camera='wrist'
    camera_name='wrist_cam'
    model_builder=staticmethod(so101_model)
    home_native=np.array([-11.00079,-15.01149,14.49583,64.74423,4.54356,50.76047])

    def __init__(self, *, images=False, deadline_s=20., render_mode=None,variation='nominal'):
        xml, self.manifest = self.model_builder()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        from .variation import EpisodeVariation
        self.variation=EpisodeVariation(self.model,self.camera_name);self.variation_mode=variation
        self.images = images
        self.renderer = None
        self.physics_observer = None  # optional measurement after EVERY physics substep
        self.render_mode = render_mode
        self.dt = 1/self.fps
        self.deadline_s = deadline_s
        self.joints = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
        self.ranges = self.model.actuator_ctrlrange.copy()
        low, high = self.from_radians(self.ranges[:,0]), self.from_radians(self.ranges[:,1])
        self.action_space = gym.spaces.Box(np.minimum(low,high).astype(np.float32), np.maximum(low,high).astype(np.float32))
        features = {'state': gym.spaces.Box(-np.inf,np.inf,(6,),dtype=np.float32)}
        if images:
            features['rgb'] = gym.spaces.Box(0,255,(480,640,3),dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(features)
        self.reset(seed=0)

    def from_radians(self, q):
        result = np.rad2deg(np.asarray(q)).copy()
        result[5] = (q[5]-self.ranges[5,0])/np.ptp(self.ranges[5])*100
        return result

    def to_radians(self, q):
        result = np.deg2rad(np.asarray(q)).copy()
        result[5] = self.ranges[5,0]+q[5]/100*np.ptp(self.ranges[5])
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:6] = self.to_radians(self.home_native)
        free=int(self.model.joint('cube_free').qposadr[0])
        if free==7:self.data.qpos[6]=self.data.qpos[5]
        self.data.qpos[free:free+3] = [self.np_random.uniform(.22,.26),self.np_random.uniform(-.08,-.04),.018]
        self.data.qpos[free+3:free+7] = [1,0,0,0]
        self.data.ctrl[:] = self.data.qpos[:6]
        self.filtered = self.data.ctrl.copy()
        self.pending = self.data.ctrl.copy()
        self.elapsed = 0.
        self.stable = 0.
        self.phase = 'approach'
        self.steps = 0
        self.lag_s = float(self.np_random.uniform(.025,.06))
        self.policy_state = {'chunk':[], 'index':0}
        self.sensor_queue = []
        self.variation_mode=(options or {}).get('variation',self.variation_mode)
        self.episode_parameters=self.variation.apply(int(self.np_random.integers(0,2**31)),self.variation_mode)
        mujoco.mj_forward(self.model,self.data)
        return self.observation(), self.info()

    def observation(self):
        # No object coordinates, contacts, task phase or target pose for ACT.
        result = {'state':self.from_radians(self.data.qpos[:6]).astype(np.float32)}
        if self.images:
            result['rgb'] = self.render()
        return result

    def neutral_action(self):
        return self.observation()['state'].tolist()

    def applied_command(self):
        return self.from_radians(self.data.ctrl).tolist()

    def teacher_state(self):
        # Privileged PPO only: cube orientation/velocity and transport/filter
        # state matter to the next transition; ACT never receives this vector.
        return np.concatenate([self.data.qpos,self.data.qvel,
            self.data.site('gripperframe').xpos,self.data.body('target_box').xpos,
            self.filtered,self.pending,[self.lag_s,self.stable,self.elapsed/self.deadline_s,
            self.model.body_mass[self.model.body('task_cube').id],self.model.geom_friction[self.model.geom('cube').id,0]]]).astype(np.float32)

    def gripper_is_open(self):
        return self.data.qpos[5]>.45

    def info(self):
        cube = self.data.body('task_cube').xpos.copy()
        target = self.data.body('target_box').xpos.copy()
        released = self.gripper_is_open() and np.linalg.norm(self.data.site('gripperframe').xpos-cube) > .055
        inside = abs(cube[0]-target[0])<.034 and abs(cube[1]-target[1])<.024 and .018<cube[2]<.04
        free_velocity=int(self.model.joint('cube_free').dofadr[0])
        stationary = np.linalg.norm(self.data.qvel[free_velocity:free_velocity+3]) < .025
        return {'domain':'sim','profile':self.profile,'model':self.manifest,
                'phase':self.phase,'inside_box':bool(inside),'released':bool(released),
                'stationary':bool(stationary),'stable_s':self.stable,'success':self.stable>=.5,
                'elapsed_s':self.elapsed, 'calibration':'uncalibrated','variation':self.episode_parameters}

    def step(self, action):
        a = np.asarray(action,dtype=float)
        if a.shape!=(6,) or not np.isfinite(a).all():
            raise ValueError('Six finite joint targets required')
        target = self.to_radians(np.clip(a,self.action_space.low,self.action_space.high))
        end = (self.steps+1)*self.dt
        while self.data.time < end-1e-10:
            # Real dynamics: filtered controls, finite motor forces and contacts; never overwrite qpos here.
            self.filtered += min(1.,self.model.opt.timestep/self.lag_s)*(self.pending-self.filtered)
            self.data.ctrl[:] = self.filtered
            mujoco.mj_step(self.model,self.data)
            if self.physics_observer is not None:
                self.physics_observer(self.model,self.data)
        self.pending = target  # one command-period transport delay
        self.steps += 1
        self.elapsed = self.data.time
        info = self.info()
        self.stable = self.stable+self.dt if info['inside_box'] and info['released'] and info['stationary'] else 0.
        cube = self.data.body('task_cube').xpos
        ee = self.data.site('gripperframe').xpos
        target_xyz = self.data.body('target_box').xpos + [0,0,.025]
        lifted = cube[2] > .055
        self.phase = 'place' if lifted else 'approach'
        reward = -float(np.linalg.norm(ee-cube)) + float(lifted)*.5 - float(np.linalg.norm(cube-target_xyz))
        info = self.info()
        success = info['success']
        deadline = self.steps*self.dt+1e-9>=self.deadline_s
        fallen = cube[2]<-.02 or np.linalg.norm(cube[:2])>1
        terminated = success or fallen
        truncated = deadline and not terminated
        info['outcome'] = 'success' if success else 'failure' if terminated or truncated else None
        info['termination_reason'] = 'stable_released_in_box' if success else 'task_deadline' if truncated else 'object_lost' if fallen else None
        return self.observation(),reward+float(success)*10,terminated,truncated,info

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model,spec))
        mujoco.mj_getState(self.model,self.data,state,spec)
        result = {'physics':state,'rng':copy.deepcopy(self.np_random.bit_generator.state),
                  'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                  'filtered':self.filtered.copy(),'pending':self.pending.copy(),
                  'elapsed':self.elapsed,'stable':self.stable,'phase':self.phase,'steps':self.steps,
                  'lag_s':self.lag_s,'policy_state':copy.deepcopy(self.policy_state),
                  'sensor_queue':copy.deepcopy(self.sensor_queue),'model_manifest':self.manifest.copy()}
        result['episode_variation']=self.variation.snapshot()
        # Torch is optional in physics-only processes, but capture it when the policy uses it.
        import sys
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            result['torch_rng'] = torch.get_rng_state().clone()
            if torch.cuda.is_initialized():
                result['cuda_rng'] = torch.cuda.get_rng_state_all()
        return result

    def restore(self, snapshot):
        if snapshot['model_manifest'] != self.manifest:
            raise ValueError('Snapshot model mismatch')
        self.variation.restore(snapshot['episode_variation'])
        self.episode_parameters=copy.deepcopy(self.variation.realization)
        self.variation_mode=self.episode_parameters['mode']
        mujoco.mj_setState(self.model,self.data,snapshot['physics'],mujoco.mjtState.mjSTATE_INTEGRATION)
        self.np_random.bit_generator.state = copy.deepcopy(snapshot['rng'])
        random.setstate(snapshot['python_rng']); np.random.set_state(snapshot['numpy_rng'])
        for key in ('filtered','pending','elapsed','stable','phase','steps','lag_s','policy_state','sensor_queue'):
            setattr(self,key,copy.deepcopy(snapshot[key]))
        if 'torch_rng' in snapshot:
            import torch
            torch.set_rng_state(snapshot['torch_rng'])
            if 'cuda_rng' in snapshot:
                torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
        mujoco.mj_forward(self.model,self.data)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=480,width=640)
        self.renderer.update_scene(self.data,camera=self.camera_name)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class StateTeacher(gym.Wrapper):
    """Explicit privileged-state PPO wrapper; never passed to ACT recording."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,env.teacher_state().shape,dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.,1.,env.action_space.shape,dtype=np.float32)

    def reset(self,**kwargs):
        _,info = self.env.reset(**kwargs)
        return self.env.teacher_state(),info

    def step(self,action):
        native = self.env.action_space.low+(np.asarray(action)+1)*.5*(self.env.action_space.high-self.env.action_space.low)
        _,r,t,tr,info = self.env.step(native)
        return self.env.teacher_state(),r,t,tr,info
