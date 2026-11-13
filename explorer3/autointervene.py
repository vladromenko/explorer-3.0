"""AutoIntervene-adapted: port of arXiv:2608.07065v1, section III, Eqs 1–6.

Uses frozen ACT features instead of DINOv3. One separate memory per profile and
checkpoint; never treats a risk score as the causal value of human assistance.
"""
import copy
import hashlib
import json
from collections import deque
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Settings:
    prefix:int=8
    visual_neighbors:int=16
    action_neighbors:int=4
    trajectories:int=2
    window:int=32
    advance_actions:int=4
    risk_history:int=3
    reject_persistence:int=3
    accept_persistence:int=3
    tail_probability:float=.05

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name!='tail_probability' and (not isinstance(value,int) or value<1):raise ValueError('Positive integer setting required')
        if not 0<self.tail_probability<.5:raise ValueError('Invalid calibration quantile')
        if self.action_neighbors>self.visual_neighbors:raise ValueError('M must not exceed K')


class Memory:
    def __init__(self,trajectories,*,profile,policy_id,settings=Settings(),groups=None):
        self.profile=profile;self.policy_id=policy_id;self.settings=settings
        self.sources=[];features=[];actions=[];trajectory_ids=[];positions=[]
        for trajectory in trajectories:
            if trajectory['outcome']!='success' or trajectory['role']!='memory':raise ValueError('Successful training references required')
            if trajectory['policy_id']!=policy_id or trajectory['profile']!=profile:raise ValueError('Feature/checkpoint/profile mismatch')
            self.sources.append({k:trajectory[k] for k in ('id','session','scene')})
            embedding=np.asarray(trajectory['features'],dtype=np.float64);action=np.asarray(trajectory['actions'],dtype=np.float64)
            if embedding.ndim!=3 or action.ndim!=2 or len(embedding)!=len(action):raise ValueError('Frame-aligned features/actions required')
            count=len(action)-settings.prefix+1
            if count<1:raise ValueError('Trajectory shorter than retained action prefix')
            for position in range(count):
                features.append(embedding[position]);actions.append(action[position:position+settings.prefix])
                trajectory_ids.append(len(self.sources)-1);positions.append(position)
        if not features:raise ValueError('Empty support memory')
        self.features=np.asarray(features);self.actions=np.asarray(actions)
        if not np.isfinite(self.features).all() or not np.isfinite(self.actions).all():raise ValueError('Nonfinite references')
        norms=np.linalg.norm(self.features,axis=-1,keepdims=True)
        if np.any(norms<1e-12):raise ValueError('Empty visual embedding')
        self.features/=norms
        self.trajectory=np.array(trajectory_ids);self.position=np.array(positions)
        self.std=np.maximum(self.actions.std(axis=(0,1)),1e-6)
        self.groups=groups or [list(range(self.actions.shape[-1]))]
        if sorted(i for group in self.groups for i in group)!=list(range(self.actions.shape[-1])):
            raise ValueError('Action groups must partition all channels')
        self.last={i:int(self.position[self.trajectory==i].max()) for i in range(len(self.sources))}
        descriptor={'profile':profile,'policy_id':policy_id,'sources':self.sources,'settings':asdict(settings),'groups':self.groups}
        self.ident=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()+self.features.tobytes()+self.actions.tobytes()).hexdigest()
        for array in (self.features,self.actions,self.std,self.trajectory,self.position):array.flags.writeable=False

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream,**{name:getattr(self,name) for name in ('features','actions','std','trajectory','position')})
        path.with_suffix('.json').write_text(json.dumps({'profile':self.profile,'policy_id':self.policy_id,
            'settings':asdict(self.settings),'groups':self.groups,'sources':self.sources,'last':self.last,'memory_id':self.ident,
            'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))

    @classmethod
    def load(cls,path):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['npz_sha256']:raise ValueError('Support memory changed on disk')
        value=cls.__new__(cls)
        for name in ('profile','policy_id','groups','sources'):setattr(value,name,meta[name])
        value.settings=Settings(**meta['settings']);value.last={int(k):v for k,v in meta['last'].items()};value.ident=meta['memory_id']
        with np.load(path,allow_pickle=False) as arrays:
            for name in ('features','actions','std','trajectory','position'):
                array=arrays[name];array.flags.writeable=False;setattr(value,name,array)
        descriptor={'profile':value.profile,'policy_id':value.policy_id,'sources':value.sources,'settings':asdict(value.settings),'groups':value.groups}
        actual=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()+value.features.tobytes()+value.actions.tobytes()).hexdigest()
        if actual!=value.ident:raise ValueError('Support memory metadata changed')
        value.last={i:int(value.position[value.trajectory==i].max()) for i in range(len(value.sources))}
        return value

    def similarities(self,features):
        query=np.asarray(features,dtype=float)
        if query.shape!=self.features.shape[1:] or not np.isfinite(query).all():raise ValueError('Wrong visual feature space')
        norm=np.linalg.norm(query,axis=-1,keepdims=True)
        if np.any(norm<1e-12):raise ValueError('Empty visual query')
        return np.min(np.sum(self.features*(query/norm),axis=-1),axis=-1)


class SupportMonitor:
    def __init__(self,memory,thresholds=None):
        self.memory=memory;self.cfg=memory.settings;self.thresholds=thresholds
        self.calibration_id=hashlib.sha256(json.dumps(thresholds,sort_keys=True).encode()).hexdigest()
        if thresholds and (thresholds['memory_id']!=memory.ident or thresholds['policy_id']!=memory.policy_id):
            raise ValueError('Rebuild memory and calibration after changing policy/data/settings')
        self.reset()

    def reset(self,mode='operator'):
        if mode not in {'policy','operator'}:raise ValueError('Unknown authority')
        self.mode=mode;self.windows={};self.risks=deque(maxlen=self.cfg.risk_history)
        self.counter=0;self.executed=0

    def score(self,features,proposal,*,executed_actions=0):
        memory=self.memory;cfg=self.cfg
        proposed=np.asarray(proposal,dtype=float)[:cfg.prefix]
        if proposed.shape!=memory.actions.shape[1:] or not np.isfinite(proposed).all():raise ValueError('Complete policy action prefix required')
        similarity=memory.similarities(features)
        if self.mode=='policy' and not self.windows:
            best=[]
            for ident in memory.last:
                candidates=np.flatnonzero(memory.trajectory==ident)
                index=candidates[np.argmax(similarity[candidates])];best.append((similarity[index],ident,int(memory.position[index])))
            self.windows={ident:position for _,ident,position in sorted(best,reverse=True)[:cfg.trajectories]}
        scope=np.ones(len(similarity),dtype=bool) if self.mode=='operator' else np.zeros(len(similarity),dtype=bool)
        if self.mode=='policy':
            for ident,position in self.windows.items():
                scope|=(memory.trajectory==ident)&(memory.position>=position)&(memory.position<position+cfg.window)
        candidates=np.flatnonzero(scope)
        if not len(candidates):return {'support':-1.,'risk':float('inf'),'exhausted':True,'references':[]}
        nearest=candidates[np.argsort(-similarity[candidates],kind='stable')[:cfg.visual_neighbors]]
        groups=[]
        for channels in memory.groups:
            difference=(proposed[:,channels]-memory.actions[nearest][:,:,channels])/memory.std[channels]
            distance=np.linalg.norm(difference.reshape(len(nearest),-1),axis=1)
            selected=np.argsort(distance,kind='stable')[:cfg.action_neighbors]
            groups.append((float(distance[selected].mean()),nearest[selected]))
        risk,selected=max(groups,key=lambda item:item[0]);self.risks.append(risk)
        exhausted=self.mode=='policy' and all(position>=memory.last[ident] for ident,position in self.windows.items())
        if executed_actions<0:raise ValueError('Executed action count must be nonnegative')
        self.executed+=executed_actions
        if self.mode=='policy' and self.executed>=cfg.advance_actions:
            for ident,position in self.windows.items():
                matches=selected[memory.trajectory[selected]==ident]
                if len(matches):self.windows[ident]=max(position,int(memory.position[matches].max()))
            self.executed%=cfg.advance_actions
        return {'support':float(similarity[selected].mean()),'risk':float(np.mean(self.risks)),
                'instant_risk':risk,'exhausted':exhausted,'references':selected.tolist()}

    def assess(self,features,proposal,*,profile,policy_id,executed_actions=0):
        if profile!=self.memory.profile or policy_id!=self.memory.policy_id:raise ValueError('Wrong policy/profile; calibration invalid')
        if self.thresholds is None:raise ValueError('Held-out calibration required before authority decisions')
        previous=self.mode;result=self.score(features,proposal,executed_actions=executed_actions)
        threshold=self.thresholds['modes'][previous]
        accepted=result['support']>=threshold['support_min'] and result['risk']<=threshold['risk_max']
        if previous=='policy':
            self.counter=self.counter+1 if not accepted or result['exhausted'] else 0
            change=self.counter>=self.cfg.reject_persistence
        else:
            self.counter=self.counter+1 if accepted else 0
            change=self.counter>=self.cfg.accept_persistence
        if change:self.reset('operator' if previous=='policy' else 'policy')
        return {**result,'accepted':bool(accepted),'previous_mode':previous,'mode':self.mode,
                'switched':change,'reset_policy_chunks':change,'method':'AutoIntervene-adapted'}

    def snapshot(self):
        return {'memory_id':self.memory.ident,'calibration_id':self.calibration_id,'mode':self.mode,'windows':copy.deepcopy(self.windows),
                'risks':list(self.risks),'counter':self.counter,'executed':self.executed}

    def restore(self,state):
        if state['memory_id']!=self.memory.ident or state['calibration_id']!=self.calibration_id:
            raise ValueError('Monitor snapshot belongs to another memory/calibration')
        self.reset(state['mode']);self.windows={int(k):v for k,v in state['windows'].items()}
        self.risks.extend(state['risks']);self.counter=state['counter'];self.executed=state['executed']


def calibrate(memory,trajectories):
    """Only current-policy proposals on disjoint successful calibration data."""
    trajectories=list(trajectories)
    if not trajectories:raise ValueError('No held-out successful calibration trajectories')
    for trajectory in trajectories:
        if trajectory['role']!='calibration' or trajectory['outcome']!='success':raise ValueError('Calibration must be held-out successful data, not test')
        if trajectory['proposal_kind']!='policy_prediction':raise ValueError('Expert actions cannot substitute for current policy proposals')
        if trajectory['policy_id']!=memory.policy_id or trajectory['profile']!=memory.profile:raise ValueError('Calibration policy/profile mismatch')
        if any(trajectory[key]==source[key] for source in memory.sources for key in ('id','session','scene')):
            raise ValueError('Calibration overlaps policy training/memory by trajectory, session or scene')
        if len(trajectory['features'])!=len(trajectory['proposals']):raise ValueError('Unaligned calibration observations')
    modes={}
    for mode in ('policy','operator'):
        support=[];risk=[]
        for trajectory in trajectories:
            monitor=SupportMonitor(memory);monitor.reset(mode)
            for feature,proposal in zip(trajectory['features'],trajectory['proposals'],strict=True):
                value=monitor.score(feature,proposal,executed_actions=1 if mode=='policy' else 0)
                if not np.isfinite(value['risk']):raise ValueError('Empty calibration retrieval scope')
                support.append(value['support']);risk.append(value['risk'])
        if not support:raise ValueError('Calibration contains no policy proposals')
        modes[mode]={'support_min':float(np.quantile(support,memory.settings.tail_probability)),
                     'risk_max':float(np.quantile(risk,1-memory.settings.tail_probability)),
                     'evaluations':len(support)}
    return {'method':'AutoIntervene-adapted','source':'arXiv:2608.07065v1 section III',
            'memory_id':memory.ident,'profile':memory.profile,'policy_id':memory.policy_id,'modes':modes,
            'settings':asdict(memory.settings),'calibration_ids':[t['id'] for t in trajectories],
            'calibration_data_kind':sorted({t.get('data_kind','unspecified') for t in trajectories})}
