"""Official LeRobotDataset writer plus causal sidecar, one source per episode."""
import json
import time
import uuid
import numpy as np
from .paths import DATA
from .profiles import profile
from . import catalog
from .timing import validate_timing


def prepare_runtime():
    """Import the dataset stack before accepting control sessions; no hardware or training.

    Native imports can hold the GIL for seconds even in a writer thread, starving
    browser heartbeat processing during the first recording of a fresh UI process.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset

def code_version():
    import subprocess
    from .paths import ROOT
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def source_snapshot():
    import hashlib
    from .paths import ROOT
    files={}
    for folder in ('explorer3','configs'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.py','.json','.txt'}:
                files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


class EpisodeWriter:
    def __init__(self, name: str, session: str, scene: str, category='demonstration', *, domain='real'):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        cfg=profile(name)
        if domain not in {'real','sim','fixture'} or cfg.get('domain','real') != ('sim' if domain=='fixture' else domain):
            raise ValueError('Profile/domain mismatch')
        self.domain=domain
        if cfg['channels'] is None:
            raise RuntimeError('Feature contract unavailable')
        if category not in {'demonstration','rollout','correction'}:
            raise ValueError('Unknown data category')
        self.id=uuid.uuid4().hex
        self.name,self.session,self.scene,self.category=name,session,scene,category
        self.cfg=cfg
        self.root=DATA/'datasets'/name/self.id if domain=='real' else DATA/'datasets'/domain/name/self.id
        self.frames=0
        channels=cfg['channels']
        self.camera=cfg['camera']
        self.robot_id=cfg['robot']
        state_channels=cfg.get('state_channels',channels)
        size=cfg.get('image_shape',[480,640,3])
        if size is None:
            raise RuntimeError('Camera dimensions not yet accepted for this profile')
        features={'observation.state':{'dtype':'float32','shape':(len(state_channels),),'names':state_channels},
                  'action':{'dtype':'float32','shape':(len(channels),),'names':channels},
                  'observation.images.'+self.camera:{'dtype':'video','shape':tuple(size),'names':['height','width','channels']}}
        self.ds=LeRobotDataset.create(repo_id=f'local/{name}_{self.id}',fps=cfg['fps'],features=features,
                                     root=self.root,robot_type=self.robot_id,use_videos=True,
                                     image_writer_threads=2,video_backend='pyav')
        self.sidecar=(self.root/'research.jsonl').open('a')
        self.last_observation=-1
        self.initial_state=None
        self.command_source=None

    def add(self, sample: dict):
        image=sample['images'][self.camera]
        command_source=sample.get('command_source','policy' if self.category=='rollout' else 'human')
        if command_source not in ({'human','policy','safety'} if self.domain=='real' else {'human','policy','safety','scripted_expert','ppo_expert'}):
            raise ValueError('Unknown command source')
        if self.command_source is not None and self.command_source!=command_source:
            raise ValueError('Finish the segment before changing command source; action chunks cannot cross it')
        self.command_source=command_source
        if self.initial_state is None:self.initial_state=list(sample['observation.state'])
        timestamp=sample['observation_timestamp_ns']
        if timestamp<=self.last_observation:
            raise ValueError('Noncausal observation/action sequence')
        validate_timing(sample)
        self.last_observation=timestamp
        state=sample['observation.state'];action=sample['action']
        if len(state)!=len(self.cfg.get('state_channels',self.cfg['channels'])) or len(action)!=len(self.cfg['channels']):
            raise ValueError('State/action channel contract mismatch')
        if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('Nonfinite state/action')
        rgb=image['image']
        self.ds.add_frame({'observation.state':np.asarray(sample['observation.state'],dtype=np.float32),
                           'action':np.asarray(sample['action'],dtype=np.float32),
                           'observation.images.'+self.camera:rgb,'task':self.cfg['task']})
        record={k:v for k,v in sample.items() if k!='images'}
        record.update(domain=self.domain,episode_id=self.id,frame_index=self.frames,robot_id=self.robot_id,task_id=self.name,
                      session_id=self.session,scene_id=self.scene,command_source=command_source,
                      measured_response_kind=sample.get('measured_response_kind','unknown'),
                      battery_voltage=sample.get('battery_voltage'),battery_source=sample.get('battery_source','unavailable'),
                      camera={k:v for k,v in image.items() if k not in {'rgb','bgr','image'}},
                      camera_source_timestamp=image.get('source_timestamp_ns'),
                      camera_source_timestamp_status=('simulation_clock' if self.domain=='sim' else 'driver_timestamp') if 'source_timestamp_ns' in image else 'unavailable',
                      clock_domain='simulation_monotonic_ns' if self.domain=='sim' else 'fixture' if self.domain=='fixture' else 'homebrain_monotonic_ns')
        self.sidecar.write(json.dumps(record)+'\n')
        self.frames+=1

    def finish(self, outcome: str, provenance: dict):
        if self.domain=='sim' and not all(key in provenance for key in ('model','seed','controller','lineage')):
            raise ValueError('Simulation provenance incomplete')
        if outcome not in {'success','failure','aborted'}:
            raise ValueError('Unknown outcome')
        self.sidecar.flush()
        self.sidecar.close()
        if self.frames:
            self.ds.save_episode()
            self.ds.finalize()
            (self.root/'provenance.json').write_text(json.dumps({'hardware':provenance,'profile':self.cfg,'outcome':outcome,
                'category':self.category,'domain':self.domain,'synthetic':self.domain!='real','closed_at':time.time(),
                'session_id':self.session,'scene_id':self.scene,'initial_state':self.initial_state,
                'code_version':code_version(),'source_snapshot':source_snapshot(),'command_source':self.command_source},indent=2))
            catalog.add_episode(dict(id=self.id,profile=self.name,dataset=str(self.root),episode_index=0,
                session=self.session,scene=self.scene,category=self.category,outcome=outcome,frames=self.frames))
        else:
            self.ds.finalize()
        return self.id
