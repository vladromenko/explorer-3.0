"""One integrated MuJoCo world, separate task views and one human-control slot.

Local MjData objects are kinematic views for reused task/IK code, never stepped.
All dynamics, contacts and time advance in the single shared MjData. A common
solver (SO's elliptic cone/impratio, 0.5 ms) is explicit in the fleet manifest.
"""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .factory import make,EXPERTS
from .mobile import model_xml,MobileDock
from .models import so101_model,rosmaster_arm_model
from .scene import laboratory

PROFILES={'so101':'so101_pick_place_sim','rosmaster':'rosmaster_pick_place_sim','tank':'tank_dock_sim'}
ZONES={'so101':np.array([0.,-1.1,0.]),'rosmaster':np.array([0.,0.,0.]),'tank':np.array([0.,1.1,0.])}
OPTIONS={'timestep':'.0005','integrator':'implicitfast','iterations':'100',
         'ls_iterations':'20','cone':'elliptic','impratio':'10'}


def build():
    root=ET.Element('mujoco',model='Explorer shared laboratory')
    ET.SubElement(root,'option',**OPTIONS)
    ET.SubElement(root,'statistic',center='.5 0 .15',extent='2.5')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .05',solref='.001 1',solimp='.95 .99 .001 .5 2')
    ET.SubElement(world,'camera',name='overview',pos='2.7 -3.5 3.0',xyaxes='.8 .6 0 -.35 .47 .8')
    laboratory(root)
    parent=mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'))
    builders={'so101':so101_model,'rosmaster':rosmaster_arm_model,'tank':lambda:(model_xml('tank')[0],None)}
    for name,builder in builders.items():
        child=ET.fromstring(builder()[0])
        child.find('option').attrib.clear();child.find('option').attrib.update(OPTIONS)
        # One shared floor, lighting and sky. MjSpec namespaces all referenced
        # joints/meshes/equalities, including the manufacturer's mimic links.
        body=child.find('worldbody')
        for element in list(body):
            if element.tag=='light' or element.get('name')=='floor':body.remove(element)
        for element in list(child):
            if element.tag in {'keyframe','visual','statistic'}:child.remove(element)
        asset=child.find('asset')
        for element in list(asset):
            if element.tag=='texture' and element.get('type')=='skybox':asset.remove(element)
        # The new world deliberately has the same global visual settings.
        child.append(copy.deepcopy(visual))
        spec=mujoco.MjSpec.from_string(ET.tostring(child,encoding='unicode'))
        parent.attach(spec,prefix=name+'/',frame=parent.worldbody.add_frame(pos=ZONES[name]))
    model=parent.compile()
    manifest={'profile':'fleet_lab_sim','physics':'one MjModel / one integrated MjData',
        'solver':OPTIONS,'zones':{k:v.tolist() for k,v in ZONES.items()},
        'rosmaster_profile':PROFILES['rosmaster'],'variation':'nominal only',
        'xml_sha256':hashlib.sha256(parent.to_xml().encode()).hexdigest(),
        'limitations':['Tank uses original package CAD on an uncalibrated planar surrogate','no whole-body ROSMASTER control',
                      'separate work zones, no object handover','not calibrated to hardware']}
    return model,manifest


class TaskView:
    def __init__(self,name,env,model,data):
        self.name=name;self.env=env;self.model=model;self.data=data;self.origin=ZONES[name]
        self.q=[];self.v=[];self.free=[]
        for j in range(env.model.njnt):
            source=env.model.joint(j);dest=model.joint(name+'/'+source.name)
            qsize=7 if env.model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 4 if env.model.jnt_type[j]==mujoco.mjtJoint.mjJNT_BALL else 1
            vsize=6 if qsize==7 else 3 if qsize==4 else 1
            self.q.extend(range(int(dest.qposadr[0]),int(dest.qposadr[0])+qsize))
            self.v.extend(range(int(dest.dofadr[0]),int(dest.dofadr[0])+vsize))
            if qsize==7:self.free.append(int(source.qposadr[0]))
        self.q=np.array(self.q);self.v=np.array(self.v)
        self.u=np.array([model.actuator(name+'/'+env.model.actuator(i).name).id for i in range(env.model.nu)])
        self.epoch=0.;self.target=None;self.result=None;self.done=False;self.task_id=0

    def read(self,forward=False):
        e=self.env;e.data.qpos[:]=self.data.qpos[self.q];e.data.qvel[:]=self.data.qvel[self.v]
        for q in self.free:e.data.qpos[q:q+3]-=self.origin
        e.data.ctrl[:]=self.data.ctrl[self.u];e.data.time=self.data.time-self.epoch
        if forward:mujoco.mj_forward(e.model,e.data)

    def reset(self,seed):
        e=self.env;e.reset(seed=seed,options={'variation':'nominal'})
        q=e.data.qpos.copy()
        for start in self.free:q[start:start+3]+=self.origin
        self.data.qpos[self.q]=q;self.data.qvel[self.v]=e.data.qvel
        self.data.qacc_warmstart[self.v]=0.;self.data.qfrc_applied[self.v]=0.
        self.data.ctrl[self.u]=e.data.ctrl
        self.epoch=float(self.data.time);self.target=None;self.result=None;self.done=False;self.task_id+=1


class FleetWorld:
    fps=20
    profile='fleet_lab_sim'

    def __init__(self,seed=0):
        self.model,self.manifest=build();self.data=mujoco.MjData(self.model)
        self.views={name:TaskView(name,make(profile),self.model,self.data) for name,profile in PROFILES.items()}
        self.manifest['tasks']={name:v.env.manifest for name,v in self.views.items()}
        self.renderer=None;self.operator=None;self.operator_switches=0
        self.controllers={};self.last_actions={};self.lease_state={};self.scheduler_state={}
        self.reset(seed)

    def reset(self,seed=0,agent=None):
        if agent is not None and agent not in self.views:raise ValueError('Unknown robot')
        names=list(self.views) if agent is None else [agent]
        if agent is None:mujoco.mj_resetData(self.model,self.data)
        for name in names:
            self.views[name].reset(seed+list(self.views).index(name))
            self.controllers.pop(name,None);self.last_actions.pop(name,None)
        mujoco.mj_forward(self.model,self.data)
        for view in self.views.values():view.read(forward=True)
        if self.operator in names:self.operator=None

    def assign_operator(self,name):
        if name is not None and name not in self.views:raise ValueError('Unknown robot')
        if self.operator is not None and name is not None and self.operator!=name:self.operator_switches+=1
        self.operator=name  # a single slot; never an array of concurrent owners

    def demonstrate(self):
        self.controllers={name:EXPERTS[v.env.profile](v.env) for name,v in self.views.items() if not v.done}

    def stop(self):
        self.controllers.clear();self.operator=None
        for view in self.views.values():
            env=view.env
            hold=np.zeros(env.action_space.shape) if isinstance(env,MobileDock) else env.data.qpos[:6].copy()
            env.filtered=hold.copy();env.pending=hold.copy();view.target=None
        self.last_actions.clear()

    def step(self,actions=None,duration=.05):
        if not 0<duration<=1:raise ValueError('Finite step duration in (0,1] required')
        actions=actions or {}
        for name,action in actions.items():
            if name not in self.views:raise ValueError('Unknown robot')
            self.views[name].env.prepare_action(action)  # validate all before any physics
        end=self.data.time+duration
        while self.data.time<end-1e-10:
            for name,v in self.views.items():
                e=v.env;v.read()
                if not v.done and v.target is None:
                    v.read(forward=True)
                    action=actions.get(name)
                    if action is None:action=self.controllers[name].action() if name in self.controllers else e.neutral_action()
                    v.target=e.prepare_action(action);self.last_actions[name]=np.asarray(action).copy()
                e.physics_control(self.model.opt.timestep);self.data.ctrl[v.u]=e.data.ctrl
            mujoco.mj_step(self.model,self.data)
            for v in self.views.values():
                e=v.env
                if not v.done and self.data.time-v.epoch>=(e.steps+1)*e.dt-1e-10:
                    v.read(forward=True);v.result=e.complete_step(v.target);v.target=None
                    v.done=bool(v.result[2] or v.result[3])
                    if v.done:
                        self.controllers.pop(v.name,None)
                        hold=np.zeros(e.action_space.shape) if isinstance(e,MobileDock) else e.data.qpos[:6].copy()
                        e.pending=hold.copy();e.filtered=hold.copy()
            if not np.isfinite(self.data.qpos).all():raise RuntimeError('Non-finite fleet physics')
        for v in self.views.values():v.read(forward=True)
        return self.info()

    def info(self):
        return {'profile':self.profile,'time_s':float(self.data.time),'model':self.manifest,
            'operator':self.operator,'operator_switches':self.operator_switches,
            'robots':{name:{**v.env.info(),'done':v.done,'task_id':v.task_id,
                'outcome':v.result[4]['outcome'] if v.result else None,
                'controller':'scripted_expert' if name in self.controllers else 'manual_or_hold',
                'observation':v.env.observation()['state'].tolist(),
                'limits':[v.env.action_space.low.tolist(),v.env.action_space.high.tolist()]}
                for name,v in self.views.items()}}

    def render(self,agent=None):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=480,width=640)
        camera='overview' if agent is None else agent+'/'+self.views[agent].env.camera_name
        self.renderer.update_scene(self.data,camera=camera)
        return self.renderer.render().copy()

    def snapshot(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(self.model,spec));mujoco.mj_getState(self.model,self.data,state,spec)
        teachers={}
        for name,expert in self.controllers.items():
            teachers[name]={k:copy.deepcopy(v) for k,v in vars(expert).items() if k not in {'env','scratch','arm'}}
        return {'manifest':copy.deepcopy(self.manifest),'physics':state,
            'tasks':{name:{'env':v.env.snapshot(),**{k:copy.deepcopy(getattr(v,k)) for k in ('epoch','target','result','done','task_id')}} for name,v in self.views.items()},
            'teachers':teachers,'operator':self.operator,'operator_switches':self.operator_switches,
            'last_actions':copy.deepcopy(self.last_actions),'scheduler_state':copy.deepcopy(self.scheduler_state),
            'lease_state':copy.deepcopy(self.lease_state)}

    def restore(self,saved):
        if saved['manifest']!=self.manifest:raise ValueError('Fleet model mismatch')
        mujoco.mj_setState(self.model,self.data,saved['physics'],mujoco.mjtState.mjSTATE_INTEGRATION)
        for name,v in self.views.items():
            v.env.restore(saved['tasks'][name]['env'])
            for key in ('epoch','target','result','done','task_id'):setattr(v,key,copy.deepcopy(saved['tasks'][name][key]))
        mujoco.mj_forward(self.model,self.data)
        for view in self.views.values():view.read(forward=True)
        self.controllers={}
        for name,state in saved['teachers'].items():
            expert=EXPERTS[self.views[name].env.profile](self.views[name].env)
            for key,value in state.items():setattr(expert,key,copy.deepcopy(value))
            self.controllers[name]=expert
        for key in ('operator','operator_switches','last_actions','scheduler_state','lease_state'):setattr(self,key,copy.deepcopy(saved[key]))

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
        for view in self.views.values():view.env.close()
