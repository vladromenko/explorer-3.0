"""Frozen policy execution and online support monitoring in one MuJoCo world.

Physics pauses while inference runs. Native task control periods, policy queues,
monitor windows and allocator history are preserved in complete snapshots. No
optimizer, physical transport or simulator truth in learned-policy observations.
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from .fleet import FleetWorld
from .factory import EXPERTS
from .branching import RelativeClock
from .mobile import MobileDock
from . import state_codec
from explorer3.act_client import PolicyClient
from explorer3.autointervene import Memory,Settings,SupportMonitor,calibrate
from explorer3.fleet_allocation import FleetAllocator,effort_metrics
from explorer3.profiles import profile


def hold(env):
    return np.zeros(env.action_space.shape) if isinstance(env,MobileDock) else env.observation()['state'].astype(float).copy()


class FixturePolicy:
    """Explicit numerical software fixture; never registered as learned weights."""
    needs_image=False
    def __init__(self,env):
        self.env=env;self.target=hold(env);self.calls=0;self.resets=0
        self.metadata={'policy_id':'numerical-fixture-'+env.profile,'policy_type':'fixture','kind':'numerical_fixture','learned':False}
    def action(self,obs):self.calls+=1;return self.target.copy()
    def propose(self,obs):
        return {'features':[[-1.,0.]] if .05<=self.env.elapsed<.35 else [[1.,0.]],
                'chunk':[self.target.tolist()],'policy_id':self.metadata['policy_id']}
    def reset(self):self.calls=0;self.resets+=1
    def snapshot(self):return {'target':self.target.copy(),'calls':self.calls,'resets':self.resets}
    def restore(self,saved):self.target=saved['target'].copy();self.calls=saved['calls'];self.resets=saved['resets']
    def close(self):pass


class LearnedPolicy:
    needs_image=True
    def __init__(self,env,spec,log):
        self.cfg=profile(env.profile);self.client=PolicyClient(Path(spec['checkpoint']),log,device=spec.get('device','cpu'),response_timeout=60)
        self.metadata={**self.client.ready,'kind':'frozen_checkpoint','source_job':spec['source_job'],
            'checkpoint':spec['checkpoint'],'software_transfer_probe':spec.get('software_probe',False),
            'training_model_sha256':spec['training_model_sha256']}
    def action(self,obs):return self.client.action(obs,self.cfg)
    def propose(self,obs):return self.client.observation_request('propose',obs,self.cfg)
    def reset(self):self.client.request('reset')
    def snapshot(self):return self.client.request('snapshot')
    def restore(self,state):self.client.request('restore',snapshot=state)
    def close(self):self.client.close()


def fixture_monitor(env,policy):
    cfg=Settings(prefix=1,visual_neighbors=2,action_neighbors=1,trajectories=1,window=10,
        advance_actions=1000,risk_history=1,reject_persistence=2,accept_persistence=2)
    source={'id':'fixture-memory','session':'fixture-memory','scene':'fixture-memory','role':'memory',
        'outcome':'success','profile':env.profile,'policy_id':policy.metadata['policy_id'],
        'features':[[[1.,0.]],[[1.,0.]]],'actions':[policy.target.tolist(),(policy.target+.01).tolist()]}
    memory=Memory([source],profile=env.profile,policy_id=policy.metadata['policy_id'],settings=cfg)
    held={**source,'id':'fixture-held','session':'fixture-held','scene':'fixture-held','role':'calibration',
        'proposals':[[policy.target.tolist()],[policy.target.tolist()]],'proposal_kind':'policy_prediction',
        'data_kind':'numerical_fixture'}
    monitor=SupportMonitor(memory,calibrate(memory,[held]));monitor.reset('policy');return monitor


def load_monitor(spec,policy,env):
    from explorer3.vision_assets import digest
    expected=spec['support_files']
    for path,checksum in expected.items():
        if digest(path)!=checksum:raise ValueError('Support calibration changed after run preparation')
    memory=Memory.load(spec['memory']);thresholds=json.loads(Path(spec['thresholds']).read_text())
    if memory.policy_id!=policy.metadata['policy_id'] or memory.profile!=env.profile:
        raise ValueError('Support calibration belongs to another checkpoint/profile')
    if policy.metadata['policy_type']!='act':raise ValueError('ACT support is not calibrated for this policy type')
    if not thresholds.get('calibration_data_kind') or any('fixture' in x or x=='unspecified' for x in thresholds['calibration_data_kind']):
        raise ValueError('Numerical fixtures cannot authorize a learned-policy monitor')
    monitor=SupportMonitor(memory,thresholds);monitor.reset('policy');return monitor


class FleetExecution:
    def __init__(self,specs,output,*,seed=1,method='fleet_thrifty',delay_s=.1,help_s=1.,request_deadline_s=3.,help_mode='scripted_expert',gain=None,horizon_s=20.):
        if not specs or not set(specs).issubset({'so101','rosmaster','tank'}):raise ValueError('Choose at least one fleet agent')
        if help_mode not in {'scripted_expert','hold_only'}:raise ValueError('Unknown assistance mode')
        if not all(math.isfinite(x) and 0<=x<=10 for x in (delay_s,help_s,request_deadline_s)) or help_s<=0 or request_deadline_s<=0:
            raise ValueError('Finite bounded assistance budgets required')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.world=FleetWorld(seed);self.specs=copy.deepcopy(specs);self.policies={};self.monitors={};self.names=list(specs)
        self.delay_s=delay_s;self.help_s=help_s;self.deadline_s=request_deadline_s;self.help_mode=help_mode
        self.allocator=FleetAllocator(len(self.names),method,seed=seed,min_steps=max(1,math.ceil(help_s/.05)))
        self.next_allocation=0.;self.helper=None;self.helper_name=None;self.help_started=None
        self.requests={};self.last_scores={};self.executed={n:0 for n in specs};self.inflight={}
        self.counts={n:{'policy':0,'safety':0,'scripted_expert':0} for n in specs}
        self.accounting={'assignments':[],'reward':{n:0. for n in specs},'waiting_s':{n:0. for n in specs}}
        self.gain=None;self.gain_scores={};self.history={n:[] for n in specs};self.horizon_s=horizon_s
        self.before_action=None;self.events=[];self.rows=[];self.sequence=0;self.inference_seconds=0.;self.paused=False
        try:
            if gain:
                from explorer3.vision_assets import digest
                if digest(gain['path'])!=gain['sha256']:raise ValueError('Gain model changed after preparation')
                self.gain=json.loads(Path(gain['path']).read_text())
                if self.gain.get('schema')!='branch_gain_model_v1' or self.gain.get('domain')!='sim' or self.gain.get('supervision')!='scripted_expert':
                    raise ValueError('Only explicit simulated assistance models accepted')
                if self.gain['protocol']['correction_s']!=help_s:raise ValueError('Gain assistance budget mismatch')
            for name,spec in specs.items():
                env=self.world.views[name].env
                if spec['kind']=='fixture':
                    policy=FixturePolicy(env);self.policies[name]=policy
                    self.monitors[name]=fixture_monitor(env,policy)
                else:
                    allowed={env.manifest['xml_sha256'],self.world.agent_manifest(name)['xml_sha256']}
                    if spec['training_model_sha256'] not in allowed and not spec.get('software_probe',False):
                        raise ValueError('Training model mismatch; choose explicit software transfer probe or compatible weights')
                    from explorer3.training_drafts import checkpoint_hash
                    if checkpoint_hash(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('Checkpoint changed after preparation')
                    policy=LearnedPolicy(env,spec,self.output/(name+'-inference.log'));self.policies[name]=policy
                    if spec.get('memory'):self.monitors[name]=load_monitor(spec,policy,env)
            self.world.controllers.clear()
        except Exception:self.close();raise

    def event(self,kind,**values):self.events.append({'sim_time_s':float(self.world.data.time),'kind':kind,**values})

    def _release(self,reason):
        if self.helper_name is not None:
            name=self.helper_name;self.event('help_finished',agent=name,reason=reason)
            self.policies[name].reset();self.world.hold_agent(name)
        self.helper=None;self.helper_name=None;self.help_started=None;self.world.assign_operator(None)

    def _schedule(self):
        now=float(self.world.data.time)
        for name,request in self.requests.items():
            if request['status']=='pending' and now>=request['deadline']:
                request['status']='expired';self.event('request_expired',agent=name)
        if self.helper_name is not None:
            name=self.helper_name
            accepted=self.monitors[name].mode=='policy' or self.world.views[name].done
            exhausted=now-self.help_started>=self.help_s-1e-10
            if accepted or exhausted:
                self.requests[name]['status']='returned' if accepted else 'budget_exhausted'
                self._release('monitor_accepted' if accepted else 'correction_budget')
        if now+1e-10<self.next_allocation:return
        self.next_allocation+=.05
        pending=[name in self.requests and self.requests[name]['status']=='pending' and
            now>=self.requests[name]['requested_at']+self.delay_s and not self.world.views[name].done for name in self.names]
        # Keep an active bounded correction in the author's single slot.
        if self.helper_name is not None:pending=[n==self.helper_name for n in self.names]
        risk=[1e12 if self.last_scores.get(n,{}).get('risk') is None and n in self.last_scores else max(0.,self.last_scores.get(n,{}).get('risk',0.)) for n in self.names]
        novelty=[max(0.,1-self.last_scores.get(n,{}).get('support',1.)) for n in self.names]
        chosen=self.allocator.allocate(risk,novelty,eligible=pending)
        if self.helper_name is None and chosen is not None:
            name=self.names[chosen];self.helper_name=name;self.help_started=now
            self.helper=EXPERTS[self.world.views[name].env.profile](RelativeClock(self.world.views[name].env))
            self.requests[name]['status']='assisting';self.policies[name].reset()
            self.event('help_started',agent=name,source=self.help_mode)
        self.world.assign_operator(self.helper_name)
        self.accounting['assignments'].append(self.allocator.assignments.tolist())
        self.world.scheduler_state=self.allocator.snapshot()

    def advance(self):
        if self.paused:raise ValueError('Execution paused; explicitly resume before advancing')
        due=[n for n in self.names if not self.world.views[n].done and self.world.views[n].target is None]
        observations={}
        for name in due:
            view=self.world.views[name];policy=self.policies[name]
            observations[name]={'state':view.env.observation()['state']}
            if policy.needs_image:observations[name]['rgb']=self.world.render(name)
            if self.gain is not None:
                from explorer3.gain_preparation import observation
                from explorer3.gain_model import predict
                compatible=self.horizon_s-self.world.data.time>=self.gain['protocol']['horizon_s']
                if compatible and policy.metadata['kind']=='frozen_checkpoint':
                    features=observation(self.history[name],view.env.elapsed,
                        view.env.action_space.high-view.env.action_space.low,self.delay_s)
                    self.gain_scores[name]={**predict(self.gain,[{'features':features}])[0],
                        'mode':'shadow_simulation','controls_assignment':False,'human_effect':False}
                else:self.gain_scores[name]={'unavailable':'insufficient horizon or numerical fixture'}
            if name in self.monitors:
                started=time.monotonic();proposal=policy.propose(observations[name]);self.inference_seconds+=time.monotonic()-started
                monitor=self.monitors[name]
                score=monitor.assess(proposal['features'],proposal['chunk'],profile=view.env.profile,
                    policy_id=proposal['policy_id'],executed_actions=self.executed[name])
                self.executed[name]=0
                # No calibrated probability or causal gain is invented from this distance.
                if not math.isfinite(score['risk']):score.update(risk=None,risk_unbounded=True)
                self.last_scores[name]=score
                if score['switched']:
                    policy.reset();self.world.hold_agent(name)
                    self.event('authority_changed',agent=name,mode=score['mode'],chunk_reset=True)
                    if score['mode']=='operator':
                        now=float(self.world.data.time)
                        self.requests[name]={'status':'pending','requested_at':now,'deadline':now+self.deadline_s}
                        self.event('help_requested',agent=name)
                    elif name in self.requests:self.requests[name]['status']='returned'
        self._schedule();actions={}
        for name in due:
            env=self.world.views[name].env;monitor=self.monitors.get(name)
            if monitor is not None and monitor.mode=='operator':
                assisted=self.helper_name==name and self.help_mode=='scripted_expert'
                action=self.helper.action() if assisted else hold(env)
                source='scripted_expert' if assisted else 'safety'
            else:
                started=time.monotonic();action=self.policies[name].action(observations[name]);self.inference_seconds+=time.monotonic()-started
                source='policy'
            action=np.asarray(action,dtype=float)
            if action.shape!=env.action_space.shape or not np.isfinite(action).all():raise ValueError('Invalid policy action')
            if self.before_action is not None:self.before_action(name,action,source)
            from explorer3.gain_preparation import sample
            self.history[name].append(sample(env.profile,observations[name]['state'],action,env.elapsed))
            self.history[name]=self.history[name][-max(2,math.ceil(3/env.dt)):]
            actions[name]=action;self.inflight[name]=source
            self.rows.append({'sequence':self.sequence,'agent':name,'task_id':self.world.views[name].task_id,
                'sim_time_s':float(self.world.data.time),'source':source,'action':action.tolist(),
                'observation_state':observations[name]['state'].tolist(),'monitor':copy.deepcopy(self.last_scores.get(name)),
                'operator':self.world.operator,'domain':'sim'})
        before={n:v.env.steps for n,v in self.world.views.items()}
        before_time=float(self.world.data.time)
        self.world.step(actions,stop_at_control_boundary=True)
        for name in self.names:
            if self.monitors.get(name) is not None and self.monitors[name].mode=='operator' and self.helper_name!=name:
                self.accounting['waiting_s'][name]+=float(self.world.data.time)-before_time
            count=self.world.views[name].env.steps-before[name]
            if count:
                self.accounting['reward'][name]+=float(self.world.views[name].result[1])
                source=self.inflight[name];self.counts[name][source]+=count
                if source=='policy':self.executed[name]+=count
        self.sequence+=1
        return self.status()

    def status(self):
        return {'time_s':float(self.world.data.time),'sequence':self.sequence,'operator':self.world.operator,
            'requests':copy.deepcopy(self.requests),'monitors':copy.deepcopy(self.last_scores),'counts':copy.deepcopy(self.counts),
            'policies':{n:p.metadata for n,p in self.policies.items()},'paused':self.paused,
            'inference_wall_seconds':self.inference_seconds,'world':self.world.info(),
            'help_source':self.help_mode,'measured_human_cost':False,'gain_estimator':'shadow_simulation' if self.gain else 'unavailable; support distance is not causal gain',
            'gain_scores':copy.deepcopy(self.gain_scores)}

    def metrics(self):
        assignments=self.accounting['assignments'] or [np.zeros((len(self.names),1)).tolist()]
        correction=sum(self.counts[n]['scripted_expert']*self.world.views[n].env.dt for n in self.names)
        successes=sum(self.world.views[n].done and self.world.views[n].result[4]['outcome']=='success' for n in self.names)
        return {**effort_metrics(sum(self.accounting['reward'].values()),assignments,len(self.names),
                successes=successes,correction_s=correction,switch_s=0.,reset_s=0.),
            'reward_per_agent':copy.deepcopy(self.accounting['reward']),
            'waiting_sim_seconds':copy.deepcopy(self.accounting['waiting_s']),
            'assignment_period_sim_s':.05,'switch_reset_cost':'instantaneous simulator; no human timing',
            'reward_comparability':'raw native task rewards; compare only identical task sets and horizons'}

    def snapshot(self):
        helper=None if self.helper is None else {k:copy.deepcopy(v) for k,v in vars(self.helper).items() if k not in {'env','scratch','arm'}}
        return {'world':self.world.snapshot(),'policies':{n:p.snapshot() for n,p in self.policies.items()},
            'monitors':{n:m.snapshot() for n,m in self.monitors.items()},'allocator':self.allocator.snapshot(),
            'identity':self.specs,'gain_identity':self.gain,'helper':helper,'helper_clock_start':self.helper.env.start if self.helper is not None else None,**{k:copy.deepcopy(getattr(self,k)) for k in
            ('next_allocation','helper_name','help_started','requests','last_scores','executed','inflight','counts','accounting','history','gain_scores','events','rows','sequence','paused')}}

    def restore(self,saved,*,interactive=False):
        if saved['identity']!=self.specs:raise ValueError('Fleet policy set changed')
        if saved.get('gain_identity')!=self.gain:raise ValueError('Gain model changed')
        self.world.restore(saved['world'])
        for name,state in saved['policies'].items():self.policies[name].restore(state)
        for name,state in saved['monitors'].items():self.monitors[name].restore(state)
        self.allocator.restore(saved['allocator'])
        for key in ('next_allocation','helper_name','help_started','requests','last_scores','executed','inflight','counts','accounting','history','gain_scores','events','rows','sequence','paused'):
            setattr(self,key,copy.deepcopy(saved[key]))
        self.helper=None
        if saved['helper'] is not None:
            clock=RelativeClock(self.world.views[self.helper_name].env);clock.start=saved['helper_clock_start']
            self.helper=EXPERTS[clock.env.profile](clock)
            for key,value in saved['helper'].items():setattr(self.helper,key,copy.deepcopy(value))
        if interactive:self.paused=True

    def pause(self):self.paused=True
    def resume(self):self.paused=False

    def reset_agent(self,name,seed):
        if not self.paused:raise ValueError('Pause before resetting one agent')
        if name not in self.policies:raise ValueError('No policy for agent')
        if self.helper_name==name:self._release('agent_reset')
        self.world.reset(seed=seed,agent=name);self.policies[name].reset()
        if name in self.monitors:self.monitors[name].reset('policy')
        self.requests.pop(name,None);self.last_scores.pop(name,None);self.executed[name]=0;self.inflight.pop(name,None)
        self.history[name]=[];self.gain_scores.pop(name,None)
        self.counts[name]={k:0 for k in self.counts[name]};self.event('agent_reset',agent=name,seed=seed)

    def close(self):
        for policy in self.policies.values():policy.close()
        self.world.close()
