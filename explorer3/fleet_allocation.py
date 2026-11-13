"""Fleet-DAgger author allocations and one-operator assignment in Explorer.

The author TD score consumes externally estimated failure risk and novelty.
It is not our learned gain score, nor an implementation of C.U.R.'s critics.
"""
import copy
from types import SimpleNamespace
import numpy as np
from .vendor.fleet_dagger.random_allocation import RandomAllocation
from .vendor.fleet_dagger.TD_allocation import TDAllocation
from .vendor.fleet_dagger.assignment import Assignment


class FleetAllocator(Assignment):
    def __init__(self, robots, method='random', *, seed=1000, min_steps=20,
                 total_steps=400, action_budget=200):
        if robots<1 or method not in {'random','fleet_thrifty'}:
            raise ValueError('Invalid fleet/allocation method')
        self.exp_cfg=SimpleNamespace(num_envs=robots,num_humans=1,min_int_time=min_steps,
            num_steps=total_steps,allocation_cfg=SimpleNamespace(free_humans=True,
            action_budget=action_budget,std_dev=.1,constraints=False,
            alpha_weight=.5,combined_alpha_thresh=.15,eps=.0001))
        self.allocation=(RandomAllocation if method=='random' else TDAllocation)(self.exp_cfg)
        self.assignments=np.zeros((robots,1));self.human_timers=[0]
        self.blocked_envs=[False]*robots;self.rng=np.random.RandomState(seed)
        self.t=0;self.switches=0;self.assigned_steps=0;self.idle_robot_steps=0

    def allocate(self, risk, novelty, blocked=None, eligible=None):
        n=self.exp_cfg.num_envs
        risk=np.asarray(risk,dtype=float);novelty=np.asarray(novelty,dtype=float)
        if risk.shape!=(n,) or novelty.shape!=(n,) or not np.isfinite(np.r_[risk,novelty]).all():
            raise ValueError('One finite risk and novelty value per robot required')
        self.blocked_envs=list(blocked) if blocked is not None else [False]*n
        if len(self.blocked_envs)!=n:raise ValueError('Blocked robot count mismatch')
        metrics={'td_risk':risk,'uncertainty':novelty,'constraint_violation':self.blocked_envs,
            'assignments':self.assignments.copy(),'human_timers':self.human_timers.copy(),
            'blocked_envs':self.blocked_envs,'time':self.t}
        # Upstream uses legacy NumPy global RNG. Preserve caller state and keep an
        # independent reproducible stream for each experiment (single worker).
        previous=np.random.get_state();np.random.set_state(self.rng.get_state())
        try:priorities=self.allocation.allocate(metrics)
        finally:self.rng.set_state(np.random.get_state());np.random.set_state(previous)
        if eligible is not None:
            if len(eligible)!=n:raise ValueError('Eligible robot count mismatch')
            priorities=[i for i in priorities if eligible[i]]
            for i,allowed in enumerate(eligible):
                if not allowed and self.assignments[i].sum():
                    self.assignments[i,:]=0;self.human_timers[0]=0
        old=self.current
        self.assign_humans(priorities)
        if old is not None and self.current is not None and old!=self.current:self.switches+=1
        if self.current is not None:self.human_timers[0]+=1;self.assigned_steps+=1
        self.idle_robot_steps+=sum(bool(b) and i!=self.current for i,b in enumerate(self.blocked_envs))
        self.t+=1
        return self.current

    def config_snapshot(self):
        return {**vars(self.exp_cfg),'allocation_cfg':vars(self.exp_cfg.allocation_cfg).copy()}

    def snapshot(self):
        return {'method':type(self.allocation).__name__,'config':self.config_snapshot(),
                'allocation_history':{k:copy.deepcopy(getattr(self.allocation,k)) for k in ('risk_data','novelty_data') if hasattr(self.allocation,k)},
                'rng':self.rng.get_state(),**{k:copy.deepcopy(getattr(self,k)) for k in
                ('assignments','human_timers','blocked_envs','t','switches','assigned_steps','idle_robot_steps')}}

    def restore(self,state):
        if state['method']!=type(self.allocation).__name__ or state['config']!=self.config_snapshot():
            raise ValueError('Allocator snapshot configuration mismatch')
        if np.asarray(state['assignments']).shape!=self.assignments.shape or np.sum(state['assignments'])>1:
            raise ValueError('Invalid single-operator assignment')
        for key,value in state.get('allocation_history',{}).items():setattr(self.allocation,key,copy.deepcopy(value))
        self.rng.set_state(state['rng'])
        for key in ('assignments','human_timers','blocked_envs','t','switches','assigned_steps','idle_robot_steps'):
            setattr(self,key,copy.deepcopy(state[key]))

    @property
    def current(self):
        return int(np.argmax(self.assignments[:,0])) if self.assignments.sum() else None


def effort_metrics(reward, assignments, robots, *, successes, correction_s, switch_s, reset_s,
                   supervision='simulated_expert'):
    """ROHE: paper equation (2), distinct from plot.py's penalized-reward plot."""
    matrices=np.asarray(assignments,dtype=float)
    if matrices.ndim!=3 or matrices.shape[1]!=robots or matrices.shape[2]!=1:
        raise ValueError('Expected time × robots × one operator assignments')
    if not np.isin(matrices,[0,1]).all() or (matrices.sum(axis=1)>1).any():
        raise ValueError('Operator assigned concurrently')
    human_steps=float(np.square(matrices).sum())
    seconds=float(correction_s+switch_s+reset_s)
    return {'rohe':float(reward)/robots/(1+human_steps),
        'rohe_definition':'(M/N) * cumulative reward / (1 + sum ||assignment||_F^2)',
        'human_assignment_steps':human_steps,'completed_tasks':int(successes),
        'correction_s':correction_s,'switch_s':switch_s,'reset_s':reset_s,
        'tasks_per_supervision_minute':successes*60/seconds if seconds>0 else None,
        'supervision':supervision,'measured_human_cost':supervision=='human'}
