"""Explicit planar dynamic surrogates, driven by forces, never qpos integration.

ROSMASTER dimensions come from the saved console footprint; Tank geometry,
all masses/gains and slip are assumed until physical identification.
"""
import hashlib
import json
from explorer3.paths import ROOT
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np
from .environment import SO101PickPlace


def model_xml(robot):
    tank=robot=='tank'
    evidence=json.loads((ROOT/'configs/tank-hardware-evidence.json').read_text())['manufacturer_specs']
    mass=evidence['mass_kg'] if tank else None
    length=evidence['dimensions_m'][0]/2 if tank else .155
    width=evidence['dimensions_m'][1]/2-.02 if tank else .15
    root=ET.Element('mujoco',model=f'Explorer {robot} dynamic surrogate')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep='.002',integrator='implicitfast')
    ET.SubElement(root,'statistic',center='.5 0 .1',extent='1.3')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='0 0 3',dir='0 0 -1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .05',rgba='.25 .3 .35 1')
    base=ET.SubElement(world,'body',name='base',pos='0 0 0')
    ET.SubElement(base,'joint',name='x',type='slide',axis='1 0 0',damping='.1',limited='false')
    ET.SubElement(base,'joint',name='y',type='slide',axis='0 1 0',damping='.1',limited='false')
    ET.SubElement(base,'joint',name='yaw',type='hinge',axis='0 0 1',damping='.03',limited='false')
    if tank:
        # A dimensioned proxy, explicitly not a fabricated T1 mesh. Replace only
        # when the manufacturer's actual tank.xacro/assets have been obtained.
        ET.SubElement(base,'geom',name='chassis',type='box',pos='0 0 .091',size=f'{length} {width} .091',mass=str(mass),rgba='.16 .36 .55 1')
        for side in (-1,1):
            ET.SubElement(base,'geom',name=f'track_{side}',type='box',pos=f'0 {side*(width+.01)} .06',
                size=f'{length} .01 .06',mass='0',contype='0',conaffinity='0',rgba='.08 .1 .12 1')
        ET.SubElement(base,'camera',name='front_cam',pos=f'{length} 0 .17',xyaxes='0 -1 0 0 0 1',fovy='65')
        provenance={'source':'Hiwonder MentorPi T1 Advanced published depth-camera dimensions and mass',
            'source_url':'https://www.hiwonder.com/products/mentorpi-t1?variant=41789241655383',
            'geometry':'dimensioned bounding proxy; original T1 meshes unavailable in published branch',
            'dimensions_m':evidence['dimensions_m'],'published_mass_kg':mass,
            'unknown':['actual mass with user Pi/accessories','original Tank URDF','inertia','camera extrinsics/intrinsics']}
    else:
        from .manufacturer import native_tree
        from .vendor_rosmaster.arm_model import DCW2_ON_ARM4,REP103_OPTICAL
        from scipy.spatial.transform import Rotation
        native,provenance=native_tree(lock_arm=True)
        root.append(native.find('asset'))
        for body in native.find('worldbody'):base.append(body)
        wrist=base.find(".//body[@name='arm4']")
        rotation=Rotation.from_euler('xyz',DCW2_ON_ARM4[1]).as_matrix() @ REP103_OPTICAL @ np.diag([1,-1,-1])
        ET.SubElement(wrist,'camera',name='front_cam',pos=' '.join(str(x) for x in DCW2_ON_ARM4[0]),
            quat=' '.join(str(x) for x in Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]),fovy='60')
        mass=sum(float(x.get('mass')) for x in native.findall('.//inertial'))
    # The scene goal is shared by both chassis models.
    ET.SubElement(world,'geom',name='dock_zone',type='cylinder',pos='.9 0 .002',size='.09 .002',
        contype='0',conaffinity='0',rgba='.15 .65 .35 1')
    for geom in base.findall('.//geom'):
        if geom.get('contype')!='0':geom.set('contype','2');geom.set('conaffinity','2')
    target=ET.SubElement(world,'body',name='target',pos='1.2 0 .2')
    ET.SubElement(target,'geom',name='target_board',type='box',size='.015 .12 .2',rgba='.85 .3 .12 1',contype='2',conaffinity='2')
    ET.SubElement(target,'geom',name='target_center',type='box',pos='-.017 0 0',size='.002 .03 .05',
        contype='0',conaffinity='0',rgba='1 .9 .65 1')
    actuator=ET.SubElement(root,'actuator')
    for name,maximum in [('x',20),('y',20),('yaw',4)]:
        ET.SubElement(actuator,'motor',name=name,joint=name,ctrlrange=f'{-maximum} {maximum}')
    from .scene import laboratory
    provenance['presentation']=laboratory(root,mobile=True)
    return ET.tostring(root,encoding='unicode'),mass,provenance


class MobileDock(SO101PickPlace):
    fps=20
    camera='front'
    camera_name='front_cam'

    def __init__(self,robot='tank',*,images=False,deadline_s=20.,render_mode=None,variation='nominal'):
        if robot not in {'tank','rosmaster'}:raise ValueError('Unknown mobile model')
        self.robot=robot;self.profile=robot+'_dock_sim'
        xml,self.mass,provenance=model_xml(robot)
        self.model=mujoco.MjModel.from_xml_string(xml);self.data=mujoco.MjData(self.model)
        from .variation import EpisodeVariation
        self.variation=EpisodeVariation(self.model,self.camera_name);self.variation_mode=variation
        self.manifest={'source':'Explorer planar force-driven dynamic surrogate',
            'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'calibration':'assumed_uncalibrated',
            'manufacturer':provenance,
            'geometry_source':provenance['source'],
            'assumed':['traction/slip','command delay','gains','camera FOV','Tank proxy inertia and camera pose'],
            'limitations':'Planar force-driven chassis, fixed wheels/tracks, no wheel-floor friction because vertical DOF is constrained; object collisions retained; not a calibrated physical twin'}
        self.images=images;self.render_mode=render_mode;self.renderer=None
        self.dt=1/self.fps;self.deadline_s=deadline_s
        bounds=np.array([.2,.5] if robot=='tank' else [.3,.25,.8],dtype=np.float32)
        self.action_space=gym.spaces.Box(-bounds,bounds)
        self.height=400 if robot=='tank' else 480
        features={'state':gym.spaces.Box(-np.inf,np.inf,(2 if robot=='tank' else 6,),dtype=np.float32)}
        if images:features['rgb']=gym.spaces.Box(0,255,(self.height,640,3),dtype=np.uint8)
        self.observation_space=gym.spaces.Dict(features)
        self.goal=np.array([.9,0.,0.]);self.reset(seed=0)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=[self.np_random.uniform(-.03,.03),self.np_random.uniform(-.07,.07),self.np_random.uniform(-.12,.12)]
        self.filtered=np.zeros(self.action_space.shape);self.pending=self.filtered.copy();self.previous=self.filtered.copy()
        self.elapsed=0.;self.stable=0.;self.phase='approach';self.steps=0
        self.lag_s=float(self.np_random.uniform(.06,.14))
        self.slip=float(self.np_random.uniform(.85,1.))
        self.policy_state={'chunk':[],'index':0};self.sensor_queue=[]
        self.variation_mode=(options or {}).get('variation',self.variation_mode)
        self.episode_parameters=self.variation.apply(int(self.np_random.integers(0,2**31)),self.variation_mode)
        mujoco.mj_forward(self.model,self.data)
        return self.observation(),self.info()

    def body_velocity(self):
        yaw=self.data.qpos[2];c,s=np.cos(yaw),np.sin(yaw)
        vx,vy,w=self.data.qvel
        return np.array([c*vx+s*vy,-s*vx+c*vy,w])

    def observation(self):
        state=self.previous.copy() if self.robot=='tank' else np.r_[self.body_velocity(),self.previous]
        obs={'state':state.astype(np.float32)}
        if self.images:obs['rgb']=self.render()
        return obs

    def teacher_state(self):
        return np.r_[self.data.qpos,self.data.qvel,self.goal,self.filtered,self.pending,self.previous,
                     self.slip,self.lag_s,self.stable,self.elapsed/self.deadline_s].astype(np.float32)

    def neutral_action(self):return np.zeros(self.action_space.shape).tolist()

    def applied_command(self):return self.filtered.tolist()

    def info(self):
        error=self.goal-self.data.qpos
        yaw_error=np.arctan2(np.sin(error[2]),np.cos(error[2]))
        stationary=np.linalg.norm(self.data.qvel[:2])<.025 and abs(self.data.qvel[2])<.05
        return {'domain':'sim','profile':self.profile,'distance_m':float(np.linalg.norm(error[:2])),
                'yaw_error_rad':float(yaw_error),'stationary':bool(stationary),'stable_s':self.stable,
                'success':self.stable>=.5,'elapsed_s':self.elapsed,'calibration':'assumed_uncalibrated',
                'phase':self.phase,'model':self.manifest,'variation':self.episode_parameters}

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=self.action_space.shape or not np.isfinite(action).all():raise ValueError('Invalid velocity command')
        target=np.clip(action,self.action_space.low,self.action_space.high)
        end=(self.steps+1)*self.dt
        while self.data.time<end-1e-10:
            self.filtered+=min(1.,self.model.opt.timestep/self.lag_s)*(self.pending-self.filtered)
            vx,vy,w=(self.filtered[0],0.,self.filtered[1]) if self.robot=='tank' else self.filtered
            yaw=self.data.qpos[2];c,s=np.cos(yaw),np.sin(yaw)
            desired=np.array([c*vx-s*vy,s*vx+c*vy,w])*self.slip
            response=np.array([self.mass/.12,self.mass/.12,.12/.10])*(desired-self.data.qvel)
            self.data.ctrl[:]=np.clip(response,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
            mujoco.mj_step(self.model,self.data)
        self.pending=target;self.previous=target.copy();self.steps+=1;self.elapsed=self.data.time
        info=self.info()
        in_zone=info['distance_m']<.08 and abs(info['yaw_error_rad'])<.15
        self.stable=self.stable+self.dt if in_zone and info['stationary'] else 0.
        self.phase='settle' if in_zone else 'approach'
        info=self.info();success=info['success'];deadline=self.steps*self.dt+1e-9>=self.deadline_s
        lost=np.linalg.norm(self.data.qpos[:2])>2.5
        terminated=success or lost;truncated=deadline and not terminated
        info['outcome']='success' if success else 'failure' if terminated or truncated else None
        info['termination_reason']='stable_in_dock_zone' if success else 'task_deadline' if truncated else 'workspace_exit' if lost else None
        reward=-info['distance_m']-.1*abs(info['yaw_error_rad'])-.01*np.linalg.norm(target)+10*float(success)
        return self.observation(),float(reward),terminated,truncated,info

    def snapshot(self):
        snapshot=super().snapshot();snapshot.update(previous=self.previous.copy(),slip=self.slip,goal=self.goal.copy());return snapshot

    def restore(self,snapshot):
        super().restore(snapshot);self.previous=snapshot['previous'].copy();self.slip=snapshot['slip'];self.goal=snapshot['goal'].copy()

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=self.height,width=640)
        self.renderer.update_scene(self.data,camera=self.camera_name)
        return self.renderer.render().copy()


class DockExpert:
    def __init__(self,env):self.env=env;self.phase='approach'

    def action(self):
        env=self.env;dx,dy=env.goal[:2]-env.data.qpos[:2];yaw=env.data.qpos[2]
        error_yaw=np.arctan2(np.sin(-yaw),np.cos(-yaw))
        distance=np.hypot(dx,dy)
        if env.robot=='tank':
            heading=np.arctan2(dy,dx)-yaw
            heading=np.arctan2(np.sin(heading),np.cos(heading))
            if distance<.045:v=0.;w=2*error_yaw
            else:v=min(.18,.9*distance)*max(0.,np.cos(heading));w=2*heading
            action=np.array([v,w])
        else:
            c,s=np.cos(yaw),np.sin(yaw)
            action=np.array([.9*(c*dx+s*dy),.9*(-s*dx+c*dy),2*error_yaw])
            if distance<.04:action[:2]=0
        return np.clip(action,env.action_space.low,env.action_space.high)
