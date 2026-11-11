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
        features={'observation.state':{'dtype':'float32','shape':(len(channels),),'names':channels},
                  'action':{'dtype':'float32','shape':(len(channels),),'names':channels},
                  'observation.images.wrist':{'dtype':'video','shape':(480,640,3),'names':['height','width','channels']}}
        self.ds=LeRobotDataset.create(repo_id=f'local/{name}_{self.id}',fps=cfg['fps'],features=features,
                                     root=self.root,robot_type='so101_follower',use_videos=True,
                                     image_writer_threads=2,video_backend='pyav')
        self.sidecar=(self.root/'research.jsonl').open('a')
        self.last_observation=-1

    def add(self, sample: dict):
        image=sample['images']['wrist']
        timestamp=sample['observation_timestamp_ns']
        if timestamp<=self.last_observation or sample['action_send_start_ns']<timestamp:
            raise ValueError('Noncausal observation/action sequence')
        self.last_observation=timestamp
        rgb=image['image']
        self.ds.add_frame({'observation.state':np.asarray(sample['observation.state'],dtype=np.float32),
                           'action':np.asarray(sample['action'],dtype=np.float32),
                           'observation.images.wrist':rgb,'task':self.cfg['task']})
        record={k:v for k,v in sample.items() if k!='images'}
        record.update(episode_id=self.id,frame_index=self.frames,robot_id='so101_follower',task_id=self.name,
                      session_id=self.session,scene_id=self.scene,command_source='human' if self.category!='rollout' else 'policy',
                      measured_response_kind='measured_joint_position',battery_voltage=None,battery_source='unavailable',
                      camera={k:v for k,v in image.items() if k not in {'rgb','bgr','image'}},
                      camera_source_timestamp=None,camera_source_timestamp_status='unavailable',
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
