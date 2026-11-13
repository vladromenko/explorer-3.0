"""Temporary simulator transport into the existing official LeRobot writer.

The spool is not a training dataset. Only import_episode creates catalogued data.
"""
import hashlib
import marshal
import json
import uuid
from pathlib import Path
import cv2
from explorer3.paths import DATA


class SimCapture:
    def __init__(self, env, seed, source='human'):
        self.ident=uuid.uuid4().hex
        self.root=DATA/'staging/sim'/self.ident
        self.root.mkdir(parents=True)
        self.count=0
        self.finished=False
        self.rows=(self.root/'samples.jsonl').open('w')
        self.meta={'profile':env.profile,'camera':env.camera,'domain':'sim','seed':int(seed),'model':env.manifest,
                   'episode_parameters':env.episode_parameters,
                   'controller':{'name':'native_position_lag' if 'pick_place' in env.profile else 'planar_force_velocity_controller','lag_s':env.lag_s,'delay_frames':1},
                   'lineage':self.ident,'command_source':source,'session':self.ident,
                   'scene':f"{env.profile}-seed-{seed}-{env.manifest['xml_sha256']}",
                   'calibration':'uncalibrated','privileged_state_in_policy_observation':False,
                   'loaded_controller_sha256':hashlib.sha256(b''.join(marshal.dumps(getattr(env,name).__func__.__code__) for name in ('step','reset','observation','render'))).hexdigest()}

    def add(self,env,action):
        if self.finished:raise RuntimeError('Capture already finished')
        timestamp=round(env.elapsed*1e9)+1
        image=env.render()
        filename=f'{self.count:06d}.png'
        if not cv2.imwrite(str(self.root/filename),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):
            raise OSError('Unable to save simulator camera')
        sample={'observation.state':env.observation()['state'].tolist(),'action':list(map(float,action)),
                'observation_timestamp_ns':timestamp,'action_send_start_ns':timestamp,
                'action_send_end_ns':timestamp,'action_available_ns':timestamp,'action_acknowledged':False,
                'action_time_kind':'simulation_control_submission',
                'measured_response_kind':('previous_command_estimate_no_wheel_feedback' if env.profile=='tank_dock_sim' else 'simulated_base_velocity' if env.profile=='rosmaster_dock_sim' else 'simulated_joint_position'),'state_valid':True,
                'command_source':self.meta['command_source'],'image_path':filename,
                'camera':{'timestamp_ns':timestamp,'source_timestamp_ns':timestamp,'sequence':self.count},
                'applied_command':env.applied_command(),
                'task_phase':env.phase,'seed':self.meta['seed'],'lineage':self.ident}
        self.rows.write(json.dumps(sample)+'\n')
        self.count+=1

    def finish(self,outcome,reason):
        if self.finished:return self.root
        if outcome not in {'success','failure','aborted'}:raise ValueError('Unknown capture outcome')
        try:self.rows.flush()
        finally:self.rows.close()
        self.meta.update(outcome=outcome,termination_reason=reason,frames=self.count)
        temporary=self.root/'manifest.json.tmp'
        temporary.write_text(json.dumps(self.meta,indent=2))
        temporary.replace(self.root/'manifest.json')
        self.finished=True
        return self.root
