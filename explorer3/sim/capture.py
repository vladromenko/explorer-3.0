"""Temporary simulator transport into the existing official LeRobot writer.

The spool is not a training dataset. Only import_episode creates catalogued data.
"""
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
        self.rows=(self.root/'samples.jsonl').open('w')
        self.meta={'profile':env.profile,'camera':env.camera,'domain':'sim','seed':int(seed),'model':env.manifest,
                   'controller':{'name':'native_position_lag','lag_s':env.lag_s,'delay_frames':1},
                   'lineage':self.ident,'command_source':source,'session':self.ident,
                   'scene':f"{env.profile}-seed-{seed}-{env.manifest['xml_sha256']}",
                   'calibration':'uncalibrated','privileged_state_in_policy_observation':False}

    def add(self,env,action):
        timestamp=round(env.elapsed*1e9)+1
        image=env.render()
        filename=f'{self.count:06d}.png'
        if not cv2.imwrite(str(self.root/filename),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):
            raise OSError('Unable to save simulator camera')
        sample={'observation.state':env.observation()['state'].tolist(),'action':list(map(float,action)),
                'observation_timestamp_ns':timestamp,'action_send_start_ns':timestamp,
                'action_send_end_ns':timestamp,'action_available_ns':timestamp,'action_acknowledged':False,
                'action_time_kind':'simulation_control_submission',
                'measured_response_kind':'simulated_joint_position','state_valid':True,
                'command_source':self.meta['command_source'],'image_path':filename,
                'camera':{'timestamp_ns':timestamp,'source_timestamp_ns':timestamp,'sequence':self.count},
                'applied_command':env.from_radians(env.data.ctrl).tolist(),
                'task_phase':env.phase,'seed':self.meta['seed'],'lineage':self.ident}
        self.rows.write(json.dumps(sample)+'\n')
        self.count+=1

    def finish(self,outcome,reason):
        self.rows.flush(); self.rows.close()
        self.meta.update(outcome=outcome,termination_reason=reason,frames=self.count)
        (self.root/'manifest.json').write_text(json.dumps(self.meta,indent=2))
        return self.root
