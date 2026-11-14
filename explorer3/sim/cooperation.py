"""S1 uses FleetWorld physics and robot RGB; oracle outcome is kept separate.

The fixed observation pose is the existing ArmModel.look_at_ik solution for the
known simulated dock marker. It is NOT a generalized target-seeking policy.
"""
import json
import sys
import time
from pathlib import Path
import cv2
import mujoco
import numpy as np
from .fleet import FleetWorld
from .mobile import DockExpert

OBSERVE_POSE=[46.76362916,104.44152446,23.57243078,44.75276349,90.,30.]


def marker(rgb):
    # Conservative, declared color/shape baseline in the fixed nominal scene.
    # Only image pixels: no segmentation IDs, object coordinates or task state.
    data=rgb.astype(float);r,g,b=data[:,:,0],data[:,:,1],data[:,:,2]
    mask=((r>45)&(r>1.7*g)&(r>2*b)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    matches=[]
    for index in range(1,count):
        x,y,w,h,area=stats[index].tolist()
        if area>=500 and h>w*1.1 and w>12 and 1<x<rgb.shape[1]-w-1:
            patch=data[y:y+h,x:x+w]
            beige=(patch[:,:,0]>60)&(patch[:,:,1]>.65*patch[:,:,0])&(patch[:,:,1]<.95*patch[:,:,0])&(patch[:,:,2]<.8*patch[:,:,1])
            if int(beige.sum())>=30:matches.append({'box_xywh':[x,y,w,h],'pixels':area,'inner_patch_pixels':int(beige.sum())})
    return {'target_visible':len(matches)==1,'detections':matches,'source':'opencv_fixed_marker_rgb_v1',
            'scope':'single nominal simulated orange dock marker; not real object identity','world_xyz':None}


class CooperationWorld:
    def __init__(self,output,seed):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.world=FleetWorld(seed);self.seed=seed;self.stage='waiting_for_observation'
        view=self.world.views['tank'];self.world.data.qpos[view.q[2]]=.9
        mujoco.mj_forward(self.world.model,self.world.data)
        for item in self.world.views.values():item.read(forward=True)
        self.initial=self.world.snapshot();self.helper_visible=False

    def camera(self,robot,label):
        image=self.world.render(robot);path=self.output/(label+'.png')
        cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        return {'robot':robot,'frame_id':robot+'/camera','image':str(path),'measurement':marker(image),
                'simulation_time_s':float(self.world.data.time)}

    def snapshot(self):
        return {'scene':'fixed_shared_dock_v1','seed':self.seed,'stage':self.stage,'domain':'sim',
            'model_sha256':self.world.manifest['xml_sha256'],
            'recipient':self.camera('tank','recipient-before'),
            'donor':self.camera('rosmaster','donor-before'),
            'conditions':{'recipient_stationary':bool(np.linalg.norm(self.world.views['tank'].env.data.qvel)<1e-6),'fixed_scene':True},
            'action_provenance':'privileged scripted teacher; no learned motor policy'}

    def observe(self,scan=False):
        if self.stage!='waiting_for_observation':raise ValueError('Observation is not repeatable within this episode')
        before=self.world.views['tank'].env.data.qpos.copy()
        if scan:
            for pose in ([90.,115.,20.,40.,90.,30.],[65.,110.,20.,42.,90.,30.]):
                for _ in range(20):self.world.step({'rosmaster':pose})
        for _ in range(60):self.world.step({'rosmaster':OBSERVE_POSE})
        self.world.hold_agent('rosmaster')
        result=self.camera('rosmaster','donor-observation')
        self.helper_visible=result['measurement']['target_visible'];self.stage='observed'
        return {**result,'helper_outcome':'success' if self.helper_visible else 'failure',
            'recipient_held':bool(np.allclose(before,self.world.views['tank'].env.data.qpos,atol=1e-6)),
            'conditions':{'target_observed_by_helper':self.helper_visible,'fixed_scene':True,'recipient_stationary':True}}

    def dock(self):
        if self.stage!='observed' or not self.helper_visible:raise ValueError('Recipient blocked: fresh helper observation required')
        self.stage='executing';world=self.world;view=world.views['tank'];world.controllers['tank']=DockExpert(view.env)
        started=time.monotonic()
        for index in range(600):
            if not view.done:
                world.step()
                if index%60==0:self.camera('tank',f'recipient-{index:04d}')
        world.stop();info=view.env.info();self.stage='completed'
        outcome=view.result[4]['outcome'] if view.result else 'unknown'
        self.camera('tank','recipient-final');self.camera('rosmaster','donor-final')
        verifier={'source':'fixed_outcome_verifier','version':'shared-dock-v1','oracle_used_only_for_evaluation':True,
            'helper':'success' if self.helper_visible else 'failure','recipient':'success' if outcome=='success' else 'failure',
            'task':'success' if outcome=='success' else 'failure','raw_environment_outcome':outcome,
            'distance_m':info['distance_m'],'yaw_error_rad':info['yaw_error_rad'],'stationary':info['stationary'],
            'stable_s':info['stable_s'],'simulation_time_s':float(world.data.time),'wall_time_s':time.monotonic()-started}
        (self.output/'verifier.json').write_text(json.dumps(verifier,indent=2))
        return verifier

    def close(self):self.world.stop();self.world.close()


def serve(output,seed):
    world=CooperationWorld(output,seed)
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                if set(request)!={'command'}:raise ValueError('Only fixed commands are supported')
                command=request['command']
                if command=='snapshot':result=world.snapshot()
                elif command in ('observe','scan'):result=world.observe(scan=command=='scan')
                elif command=='dock':result=world.dock()
                elif command=='close':break
                else:raise ValueError('Unknown cooperation command')
                print(json.dumps({'ok':True,'result':result}),flush=True)
            except Exception as exc:print(json.dumps({'ok':False,'error':str(exc)}),flush=True)
    finally:world.close()


if __name__=='__main__':serve(sys.argv[1],int(sys.argv[2]))
