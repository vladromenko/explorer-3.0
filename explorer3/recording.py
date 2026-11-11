"""Official LeRobotDataset writer plus causal sidecar, one source per episode."""
import json
import time
import uuid
import numpy as np
from .paths import DATA
from .profiles import profile
from . import catalog

class EpisodeWriter:
    def __init__(self, name: str, session: str, scene: str, category='demonstration'):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        cfg=profile(name)
        if cfg['channels'] is None:
            raise RuntimeError('Feature contract unavailable')
        if category not in {'demonstration','rollout','correction'}:
            raise ValueError('Unknown data category')
        self.id=uuid.uuid4().hex
        self.name,self.session,self.scene,self.category=name,session,scene,category
        self.cfg=cfg
        self.root=DATA/'datasets'/name/self.id
        self.frames=0
        channels=cfg['channels']
        self.camera=cfg['camera']
        self.robot_id=cfg['robot']
        state_channels=cfg.get('state_channels',channels)
        size=cfg.get('image_shape',[480,640,3])
        features={'observation.state':{'dtype':'float32','shape':(len(state_channels),),'names':state_channels},
                  'action':{'dtype':'float32','shape':(len(channels),),'names':channels},
                  'observation.images.'+self.camera:{'dtype':'video','shape':tuple(size),'names':['height','width','channels']}}
        self.ds=LeRobotDataset.create(repo_id=f'local/{name}_{self.id}',fps=cfg['fps'],features=features,
                                     root=self.root,robot_type=self.robot_id,use_videos=True,
                                     image_writer_threads=2,video_backend='pyav')
        self.sidecar=(self.root/'research.jsonl').open('a')
        self.last_observation=-1

    def add(self, sample: dict):
        image=sample['images'][self.camera]
        timestamp=sample['observation_timestamp_ns']
        if timestamp<=self.last_observation or sample['action_send_start_ns']<timestamp:
            raise ValueError('Noncausal observation/action sequence')
        self.last_observation=timestamp
        rgb=image['image']
        self.ds.add_frame({'observation.state':np.asarray(sample['observation.state'],dtype=np.float32),
                           'action':np.asarray(sample['action'],dtype=np.float32),
                           'observation.images.'+self.camera:rgb,'task':self.cfg['task']})
        record={k:v for k,v in sample.items() if k!='images'}
        record.update(episode_id=self.id,frame_index=self.frames,robot_id=self.robot_id,task_id=self.name,
                      session_id=self.session,scene_id=self.scene,command_source='human' if self.category!='rollout' else 'policy',
                      measured_response_kind=sample.get('measured_response_kind','measured_joint_position'),
                      battery_voltage=sample.get('battery_voltage'),battery_source=sample.get('battery_source','unavailable'),
                      camera={k:v for k,v in image.items() if k not in {'rgb','bgr','image'}},
                      camera_source_timestamp=image.get('source_timestamp_ns'),
                      camera_source_timestamp_status='driver_timestamp' if 'source_timestamp_ns' in image else 'unavailable',
                      clock_domain='homebrain_monotonic_ns')
        self.sidecar.write(json.dumps(record)+'\n')
        self.frames+=1

    def finish(self, outcome: str, provenance: dict):
        if outcome not in {'success','failure','aborted'}:
            raise ValueError('Unknown outcome')
        self.sidecar.flush()
        self.sidecar.close()
        if self.frames:
            self.ds.save_episode()
            self.ds.finalize()
            (self.root/'provenance.json').write_text(json.dumps({'hardware':provenance,'profile':self.cfg,'outcome':outcome,
                'category':self.category,'synthetic':False,'closed_at':time.time()},indent=2))
            catalog.add_episode(dict(id=self.id,profile=self.name,dataset=str(self.root),episode_index=0,
                session=self.session,scene=self.scene,category=self.category,outcome=outcome,frames=self.frames))
        return self.id
